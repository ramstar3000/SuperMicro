// Drift estimation using entropy minimization (DME)
// 
// Note that this code also compiles without CUDA even though the extension is .cu. 
// CUDA support is implemented through a bunch of template tricks in palala.h
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021

// Note
#define _CRT_SECURE_NO_WARNINGS // strncpy

#include "palala.h"

#include "ContainerUtils.h"
#include "ThreadUtils.h"
#include "StringUtils.h"

#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <array>
#include "StringUtils.h"
#include "KDTree.h"
#include "KahanSum.h"

#include "DriftEstimation.h"

template<typename Pt>
PLL_DEVHOST float ComputeKLDivergence(Pt posA, Pt crlbA, Pt posB, Pt crlbB) {
	Pt div = crlbA / crlbB, div2 = div * div;
	Pt diff = posB - posA;
	Pt OneOverSigmaSq = 1.0f / (crlbB * crlbB);
	float trace = div2.sum();

	return (trace - log(div2.prod()) + ((diff * diff) * OneOverSigmaSq).sum() - posA.size) * 0.5f;
};

template<typename Pt>
static Pt GetMeanCrlb(const Pt* crlb, int numspots)
{
	Pt sum = {};
	for (int i = 0; i < numspots; i++)
		sum += crlb[i];

	sum /= numspots;
	return sum;
}

class IDriftEstimator {
public:
	virtual ~IDriftEstimator() {}
	virtual int NumDims() = 0;
	virtual int NumFrames() = 0;
	virtual void GetDriftEstimate(float* data) = 0;
	virtual bool Step(double& score) = 0;

	const std::string& Status() { return status;  }

protected:
	std::string status;
};

template<int D>
class LocalizationDriftEstimator : public IDriftEstimator {
public:
	std::vector<int> framenum;
	std::vector <int> sifList, sifStart, sifCount; // sif: spots-in-frame. Stored in such a way it can be easily copied to cuda

	typedef Vector<float, D> Pt;
	std::vector<Pt> driftState;
	std::vector<Pt> undrifted;
	std::vector<Pt> positions;
	std::vector<Pt> crlb;
	Pt sigma; // in case of constant crlb
	bool useConstCRLB=true;
	int iteration = 0;
	double lastscore = 0;
	float gradientStep = 0.0f;
	int rejectCount = 0;
	int maxNeighborCount = 0;

	// the drift applied when neighborList was generated. If too far from the current drift estimate we need to rebuild it. 
	NeighborList<float, D> nblist;
	std::vector<Pt> nbListDrift, // the drift estimate when the neighbor list was last updated 
		driftPerFrame;  // current drift estimate

	std::vector<Pt> deltaDriftPerSpot;

	std::vector<Pt> prevDriftState, prevDriftStateDelta;

	bool cuda = false;
	Pt neighborSearchRange;

	virtual ~LocalizationDriftEstimator() {}

	LocalizationDriftEstimator() { }

	int NumFrames() { return (int)sifCount.size(); }
	int NumDims() { return D; }
	void GetDriftEstimate(float* dst) 
	{
		auto e = ComputeDriftPerFrame();
		std::copy(e.begin(), e.end(), (Pt*)dst);
	}


	void UpdateNeighbors()
	{
		if (!nblist.nbIndices.empty()) {
			Pt maxdiff;
			for (int i = 0; i < driftPerFrame.size(); i++) {
				Pt d = driftPerFrame[i] - nbListDrift[i];
				maxdiff = d.maximum(maxdiff);
			}
			if ( (maxdiff / neighborSearchRange).max() < 0.5f)
				return;
		}
		nbListDrift = driftPerFrame;

		//DebugPrintf("building neighbour list...\n");
		nblist.Build(undrifted, undrifted, neighborSearchRange, [&](int i, int nbIdx) {
			return framenum[i] != framenum[nbIdx]; }, maxNeighborCount);
	}

	void UpdatePositions()
	{
		driftPerFrame = ComputeDriftPerFrame();

		for (int i = 0; i < (int)undrifted.size(); i++) {
			undrifted[i] = positions[i] - driftPerFrame[framenum[i]];
		}
		UpdateNeighbors();
	}

	void Begin(const Pt* xy, const Pt* crlb, const int* spotFramenum, int numspots, bool cuda, bool useConstCRLB, 
		int maxNeighborCount, const Pt* initialDrift, float initialGradientStep)
	{
		iteration = 0;
		this->maxNeighborCount = maxNeighborCount;

		framenum.assign(spotFramenum, spotFramenum + numspots);
		positions.assign(xy, xy + numspots);
		this->cuda = cuda;
		deltaDriftPerSpot.resize(numspots);
		this->useConstCRLB = useConstCRLB;
		gradientStep = initialGradientStep;

		float searchRangeMultiplier = 3.0f;
		if (useConstCRLB) {
			sigma = *crlb;
			neighborSearchRange = sigma * searchRangeMultiplier;
		}
		else {
			neighborSearchRange = GetMeanCrlb(crlb, numspots) * searchRangeMultiplier;
			this->crlb.assign(crlb, crlb + numspots);
		}

		int nframes = *std::max_element(spotFramenum, spotFramenum + numspots) + 1;
		std::vector <std::vector<int>> spotsInFrame(nframes);
		//std::vector <int> sifList, sifStart, sifCount; // sif: spots-in-frame. Stored in such a way it can be easily copied to cuda
		for (int i = 0; i < numspots; i++) {
			spotsInFrame[spotFramenum[i]].push_back(i);
		}
		for (auto& sif : spotsInFrame) {
			sifCount.push_back((int)sif.size());
			sifStart.push_back((int)sifList.size());
			sifList.insert(sifList.end(), sif.begin(), sif.end());
		}

		undrifted.resize(numspots);

		driftState = InitializeDriftState(initialDrift);
	}

	bool Step(double& score) override
	{
		UpdatePositions();

		auto stateDeltaAndScore = ComputeDriftDelta(lastscore);
		auto driftStateDelta = stateDeltaAndScore.second;

		score = stateDeltaAndScore.first;

		if (iteration == 0 || lastscore < score) { // improvement
			prevDriftState = driftState;
			gradientStep *= 1.2f;
			this->status = SPrintf("%d. Accepting step. Score: %f. Stepsize: %e [cuda=%d, dims=%d]", iteration, score, gradientStep, cuda, D);

			lastscore = score;
			rejectCount = 0;
			prevDriftStateDelta = driftStateDelta;
		}
		else {
			if (gradientStep < 1e-16f || rejectCount == 10)
				return false;

			this->status = SPrintf("%d. Rejecting step. Score: %f. Stepsize: %e [cuda=%d, dims=%d]", iteration, score, gradientStep, cuda, D);

			// restore drift to previous position
			driftState = prevDriftState;
			driftStateDelta = prevDriftStateDelta;

			gradientStep *= 0.5f;
			rejectCount++;
		}
		iteration++;

		for (int i = 0; i < driftState.size(); i++)
			driftState[i] += driftStateDelta[i] * gradientStep;
		return true;
	}

	virtual std::vector<Pt> InitializeDriftState(const Pt* driftPerFrame) = 0;
	virtual std::vector<Pt> ComputeDriftPerFrame() = 0;
	// if lastscore > score, the returned drift delta is ignored (speed up by not computing the delta on rejected steps)
	virtual std::pair<double, std::vector<Pt>> ComputeDriftDelta(double lastScore) = 0;

	double UpdateDeltas(double lastScore)
	{
		if (useConstCRLB)
			return UpdateDeltas_ConstCRLB(lastScore);
		else
			return UpdateDeltas_PerSpotCRLB(lastScore);
	}

	double UpdateDeltas_ConstCRLB(double lastScore)
	{
		// normalization terms (Z)
		std::vector<float> norm(positions.size());
		Pt OneOverSigmaSq = 1.0f / (sigma*sigma);

		// compute normalization terms
		palala_for((int)positions.size(), cuda, PLL_FN(int i,
			float* norm, const int* nbIndices, const int* nbStart, const int* nbCount, const Pt * undrifted) {

			float sum = 1.0f; // one for i=j

			for (int j = 0; j < nbCount[i]; j++) {
				int nIdx = nbIndices[nbStart[i] + j];

				// calculate the Kullback-Leibler divergence between the 2 gaussian distributions given by the localizations (i,j):
				// ends up being simple because variances are equal here
				Pt diff = undrifted[i] - undrifted[nIdx];
				float D_kl = 0.5f * ((diff * diff) * OneOverSigmaSq).sum();
				sum += exp(-D_kl);
			}
			norm[i] = 1.0f / sum;
		}, norm,
			const_vector(nblist.nbIndices),
			const_vector(nblist.startIndices), const_vector(nblist.nbCounts),
			const_vector(undrifted));

		// norm = 1 / Z
		KahanSum<double> accum;
		for (int i = 0; i < positions.size(); i++) {
			accum += log(norm[i]);
		}
		double entropy = accum() / positions.size();
		double score = -entropy;
		if (iteration > 0 && lastScore >= score)
			return score;

		// for each spot go through all its neighbors
		palala_for((int)positions.size(), cuda, PLL_FN(int i,
			Pt * deltaDrift, const float* norm, const int* nbIndices, const int* nbStart, const int* nbCount, const Pt * undrifted, const int* framenum) {

			Pt delta = {};
			int fr = framenum[i];

			for (int j = 0; j < nbCount[i]; j++) {
				int nIdx = nbIndices[nbStart[i] + j];
				if (framenum[nIdx] == fr)
					continue;

				// in this case of equal variance, D_kl(i,j) = D_kl(j,i)
				Pt diff = undrifted[i] - undrifted[nIdx];
				float D_kl = 0.5f * ((diff * diff) * OneOverSigmaSq).sum();
				float e = exp(-D_kl);
				delta += diff * OneOverSigmaSq * ((norm[i] + norm[nIdx]) * e);
			}
			deltaDrift[i] = delta;
		}, deltaDriftPerSpot, const_vector(norm),
			const_vector(nblist.nbIndices),
			const_vector(nblist.startIndices), const_vector(nblist.nbCounts),
			const_vector(undrifted), const_vector(framenum));

		return score;
	}

	double UpdateDeltas_PerSpotCRLB(double lastScore)
	{
		// normalization terms (Z)
		std::vector<float> norm(positions.size());

		// compute normalization terms
		palala_for((int)positions.size(), cuda, PLL_FN(int i,
			float* norm, const int* nbIndices, const int* nbStart, const int* nbCount, const Pt * undrifted, const Pt * crlb) {

			float sum = 1.0f; // one for i=j

			for (int j = 0; j < nbCount[i]; j++) {
				int nIdx = nbIndices[nbStart[i] + j];
				float D_kl = ComputeKLDivergence(undrifted[i], crlb[i], undrifted[nIdx], crlb[nIdx]);
				sum += exp(-D_kl);
			}
			norm[i] = 1.0f / sum;
		}, norm,
			const_vector(nblist.nbIndices),
			const_vector(nblist.startIndices), const_vector(nblist.nbCounts),
			const_vector(undrifted), const_vector(crlb));

		// norm = 1 / Z
		KahanSum<double> accum;
		for (int i = 0; i < positions.size(); i++) {
			accum += log(norm[i]);
		}
		double entropy = accum() / positions.size();
		double score = -entropy;
		if (iteration > 0 && lastScore >= score)
			return score;

		// for each spot go through all its neighbors
		palala_for((int)positions.size(), cuda, PLL_FN(int i,
			Pt * deltaDrift, const float* norm, const int* nbIndices, const int* nbStart, const int* nbCount, const Pt * undrifted, const Pt * crlb, const int* framenum) {

			Pt delta = {};
			int fr = framenum[i];

			Pt OneOverSigmaSq_i = 1.0f / (crlb[i] * crlb[i]);

			for (int j = 0; j < nbCount[i]; j++) {
				int nIdx = nbIndices[nbStart[i] + j];
				if (framenum[nIdx] == fr)
					continue;

				Pt diff = undrifted[i] - undrifted[nIdx];
				float D_kl_1 = ComputeKLDivergence(undrifted[i], crlb[i], undrifted[nIdx], crlb[nIdx]);
				float D_kl_2 = ComputeKLDivergence(undrifted[nIdx], crlb[nIdx], undrifted[i], crlb[i]);

				Pt OneOverSigmaSq = 1.0f / (crlb[nIdx] * crlb[nIdx]);

				Pt pairDelta = diff * OneOverSigmaSq * (norm[i] * exp(-D_kl_1)) +
					diff * OneOverSigmaSq_i * (norm[nIdx] * exp(-D_kl_2));

				delta += pairDelta;

				/*
				if (i == 0 && iteration == 0) {
					DebugPrintf("VC: KLDivergence[%d,%d] = %f. KLDivergence[%d,%d] = %f. norm[%d]=%f, norm[%d]=%f, DeltaX=%f. DeltaY=%f\n",
						i, nIdx, D_kl_1, nIdx, i, D_kl_2, i, norm[i], nIdx, norm[nIdx], pairDelta[0], pairDelta[1]);
				}*/
			}
			deltaDrift[i] = delta;
		}, deltaDriftPerSpot, const_vector(norm),
			const_vector(nblist.nbIndices),
			const_vector(nblist.startIndices), const_vector(nblist.nbCounts),
			const_vector(undrifted), const_vector(crlb), const_vector(framenum));

		return score;
	}

};


template<int D>
class PerFrameMinEntropyDriftEstimator : public LocalizationDriftEstimator<D>
{
public:
	typedef LocalizationDriftEstimator<D> base; // this is required due to the C++ rules of name lookup and the base class being templated
	typedef typename LocalizationDriftEstimator<D>::Pt Pt;
	Pt sigma;

	PerFrameMinEntropyDriftEstimator() : LocalizationDriftEstimator<D>() {}

	virtual std::vector<Pt> InitializeDriftState(const Pt* driftPerFrame)
	{
		return std::vector<Pt>(driftPerFrame, driftPerFrame + base::NumFrames());
	}

	// Compute drift per frame from the drift state
	virtual std::vector<Pt> ComputeDriftPerFrame()
	{
		return std::vector<Pt>(base::driftState.begin(), base::driftState.end());
	}

	virtual std::pair<double, std::vector<Pt>> ComputeDriftDelta(double lastScore)
	{
		double score = base::UpdateDeltas(lastScore);

		std::vector<Pt> stateDelta(base::NumFrames());

		ParallelFor(base::NumFrames(), [&](int f) {
			// compute derivative for drift sx_f, sy_f
			Pt xy;
			for (int i = 0; i < base::sifCount[f]; i++) {
				int sif = base::sifList[base::sifStart[f] + i];
				xy += base::deltaDriftPerSpot[sif];
			}
			stateDelta[f] = xy;
			});

		return { score, stateDelta };
	}
};





Vector4f HermiteSplineWeights(float t) {
	float t2 = t * t, t3 = t2*t;
	Vector4f w;
	w[0] = 2 * t3 - 3.0f * t2 + 1.0f;
	w[1] = t3 - 2.0f * t2 + t;
	w[2] = -2.0f * t3 + 3.0f * t2;
	w[3] = t3 - t2;
	return w;
}

PLL_DEVHOST Vector4f CatmullRomSplineWeights(float t) {
	float t2 = t * t, t3=t2*t;
	Vector4f w = Vector4f(
		-t3 + 2 * t2 - t,
		3 * t3 - 5 * t2 + 2,
		-3 * t3 + 4 * t2 + t,
		t3 - t2
	) * 0.5f;
	return w;
}

/*
Estimate drift from localization while storing the drift as a catmull-rom spline.
*/
template<int D>
class SplineBasedMinEntropyDriftEstimator : public LocalizationDriftEstimator<D>
{
public:
	typedef LocalizationDriftEstimator<D> base;
	typedef typename LocalizationDriftEstimator<D>::Pt Pt;
	int framesPerBin;

	SplineBasedMinEntropyDriftEstimator(int framesPerBin) : framesPerBin(framesPerBin)
	{}

	virtual std::vector<Pt> InitializeDriftState(const Pt* driftPerFrame)
	{
		int nbins = (base::NumFrames() + framesPerBin - 1) / framesPerBin;
		std::vector<Pt> state(nbins);

		// set drift to mean of every bin
		for (int i = 0; i < nbins; i++) {
			int start = i * framesPerBin - framesPerBin / 2;
			int end = start + framesPerBin;
			if (start < 0) start = 0;
			if (end >= base::NumFrames()) end = base::NumFrames() - 1;
			Pt sum;
			for (int j = start; j <= end; j++) {
				sum += driftPerFrame[j];
			}
			state[i] = sum / (end - start);
		}
		return state;
	}

	// Compute drift per frame from the drift state
	virtual std::vector<Pt> ComputeDriftPerFrame()
	{
		// Evaluate the spline
		std::vector<Pt> drift(base::NumFrames());

		for (int f = 0; f < drift.size(); f++) {

			float t = f / (float)framesPerBin;
			int bin = (int)t;
			t -= bin;

			auto w = CatmullRomSplineWeights(t);

			Pt val;
			for (int j = 0; j < 4; j++) {
				int knot = std::min(std::max(0, bin + j - 1), (int)base::driftState.size() - 1);
				val += base::driftState[knot] * w[j];
			}
			drift[f] = val;
		}
		return drift;
	}

	virtual std::pair<double, std::vector<Pt>> ComputeDriftDelta(double lastScore)
	{
		double score = base::UpdateDeltas(lastScore);

		if (base::iteration > 0 && lastScore >= score)
			return { score,{} };

		std::vector< Vector< Pt, 4> > frameDeltas(base::NumFrames());
		int framesPerBin = this->framesPerBin;

		palala_for(base::NumFrames(), base::cuda, PLL_FN(int f, const int* sifCount, const int* sifList, const int* sifStart,
			const Pt * deltaDriftPerSpot, Vector<Pt, 4> * frameDeltas)
		{
			// compute derivative for drift sx_f, sy_f
			Pt xy;
			for (int i = 0; i < sifCount[f]; i++) {
				int sif = sifList[sifStart[f] + i];
				xy += deltaDriftPerSpot[sif];
			}

			// compute weights for frame f
			float t = f / (float)framesPerBin;
			int bin = (int)t;
			t -= bin;

			Vector4f w = CatmullRomSplineWeights(t);
			Vector<Pt, 4> frameDelta;
			for (int i = 0; i < 4; i++)
				frameDelta[i] = xy * w[i];

			frameDeltas[f] = frameDelta;
		}, const_vector(base::sifCount), const_vector(base::sifList), const_vector(base::sifStart), const_vector(base::deltaDriftPerSpot),
			frameDeltas);

		std::vector<Pt> stateDelta(base::driftState.size());

		for (int i = 0; i < base::NumFrames(); i++) {
			int bin = i / framesPerBin;
			for (int j = 0; j < 4; j++) {
				int knot = std::min(std::max(0, bin + j - 1), (int)stateDelta.size() - 1);
				stateDelta[knot] += frameDeltas[i][j];
			}
		}

		return { score, stateDelta };
	}
};




#define DME_3D 1
#define DME_CUDA 2
#define DME_CONSTCRLB 4

template<int D>
IDriftEstimator* DME_CreateInstance_(const float* coords_, const float* crlb_, const int* spotFramenum, int numspots,
	float* drift_, int framesPerBin, float gradientStep, float maxdrift, int flags, int maxneighbors)
{
	typedef Vector<float, D> V;
	const V* coords = (const V*)coords_;
	const V* crlb = (const V*)crlb_;

	LocalizationDriftEstimator<D>* estimator;
	bool cuda = (flags & DME_CUDA) != 0;

	if (framesPerBin == 1) {
		estimator = new PerFrameMinEntropyDriftEstimator<D>();
	}
	else {
		estimator = new SplineBasedMinEntropyDriftEstimator<D>(framesPerBin);
	}

	estimator->Begin(coords, crlb, spotFramenum, numspots, cuda, flags & DME_CONSTCRLB, maxneighbors, (const V*)drift_, gradientStep);

	return estimator;
}

CDLL_EXPORT IDriftEstimator* DME_CreateInstance(const float* coords_, const float* crlb_, const int* spotFramenum, int numspots,
	float* drift, int framesPerBin, float gradientStep, float maxdrift, int flags, int maxneighbors)
{
	try {
		IDriftEstimator * inst = 0;

		if (flags & DME_3D)
			inst = DME_CreateInstance_<3>(coords_, crlb_, spotFramenum, numspots, drift, framesPerBin, gradientStep, maxdrift, flags, maxneighbors);
		else
			inst = DME_CreateInstance_<2>(coords_, crlb_, spotFramenum, numspots, drift, framesPerBin, gradientStep, maxdrift, flags, maxneighbors);
		return inst;
	}
	catch (const std::exception& exc) {
		DebugPrintf("DME_CreateInstance Exception: %s\n", exc.what());
		return 0;
	}
}

CDLL_EXPORT int DME_Step(IDriftEstimator* estimator, char* status_msg, int status_max_length, float* score, float* drift_estimate)
{
	try {
		double score_;
		bool keepGoing = estimator->Step(score_);


		if (status_msg) {
			strncpy(status_msg, estimator->Status().c_str(), status_max_length);
		}

		if (score) {
			*score = (float)score_;
		}
		
		if (drift_estimate) {
			estimator->GetDriftEstimate(drift_estimate);
		}

		return keepGoing ? 1 : 0;
	}
	catch (const std::exception& exc) {
		DebugPrintf("DME_Step Exception: %s\n", exc.what());
		return 0;
	}
}


CDLL_EXPORT void DME_Close(IDriftEstimator* estim)
{
	try {
		delete estim;
	}
	catch (const std::exception& exc) {
		DebugPrintf("DME_Close Exception: %s\n", exc.what());
	}
}

# -*- coding: utf-8 -*-
import numpy as np
import tqdm
from .native_api import NativeAPI

from .rcc import rcc, rcc3D
        
def dme_estimate(positions, framenum, crlb, framesperbin, imgshape, 
          coarseFramesPerBin=None,
          coarseSigma=None, 
          perSpotCRLB=False,
          useCuda=True,
          display=False, # make a plot
          pixelsize=None,
          maxspots=None, 
          initializeWithRCC=10, 
          initialEstimate=None, 
          rccZoom=2,
          estimatePrecision=True,
          maxNeighbors=1000,
          maxIterations=2000,
          coarseMaxIterations=1000,
          useDebugLibrary=False,
          numframes=None,
          traces=None):
    """
    Estimate drift using minimum entropy method. Parameters:

    Required parameters: 
        
    positions: a N by K sized numpy array with all the positions, with N being the number of localizations and K the number of dimensions
    framenum: a integer numpy array of frame numbers corresponding to each localization in positions
    crlb: an N by K sized numpy array with uncertainty of positions (cramer rao lower bound from the localization code)
    framesperbin: Number of frames per spline point. Either 1 to disable spline interpolation, or >= 4 to enable cubic splines.
    imgshape: Field of view size [height, width]. Only used for computing the initial 2D estimate using RCC 
        
    Optional parameters:

    initializeWithRCC: If not None, perform RCC with 10 bins to compute an initial estimate. Pass 
    estimatePrecision: Split the dataset in two, and estimate drift on both. The difference gives an indication of estimation precision.
    display: Generate a matplotlib plot with results
    coarseFramesPerBin / coarseSigma: If not None, do a coarse initialization to prevent a local minimum. 
            coarseSigma sets an alternative 'CRLB' to smooth the optimziation landscape (typically make it 4x larger).
    pixelsize: Size of pixels in nm. If display=True, it will convert the units in the plot to nm
    maxspots: If not None, it will select the brightess spots to use and ignore the rest. Useful for large datasets > 1M spots
    initialEstimate: Initial drift estimate, replaces RCC initialization
    maxNeighbors: Limit the number of neighbors a single spot can have. 
    
    Return value:
        
    If estimatePrecision is True:
        The estimated drift of full dataset, a tuple with drifts of split dataset
    Else
        The estimated drift as numpy array
    
    """
    ndims = positions.shape[1]
    
    if numframes is None:
        numframes = np.max(framenum)+1

    initial_drift = np.zeros((numframes,ndims))
    
    with NativeAPI(useCuda, debugMode=useDebugLibrary) as dll:

        if initialEstimate is not None:
            initial_drift = np.ascontiguousarray(initialEstimate,dtype=np.float32)
            assert initial_drift.shape[1] == ndims
            
        elif initializeWithRCC:
            if type(initializeWithRCC) == bool:
                initializeWithRCC = 10
    
            posI = np.ones((len(positions),positions.shape[1]+1)) 
            posI[:,:-1] = positions
    
            if positions.shape[1] == 3:
                initial_drift = rcc3D(posI, framenum, initializeWithRCC, dll=dll, zoom=rccZoom)
            else:
                initial_drift = rcc(posI, framenum ,initializeWithRCC, dll=dll, zoom=rccZoom)[0]
            
        
        if not perSpotCRLB:
            crlb = np.mean(crlb,0)[:ndims]
            
        step = 0.000001

        splitAxis = np.argmax( np.var(positions[:,:2],0) ) # only in X or Y
        splitValue = np.median(positions[:,splitAxis])
         
                            
        maxdrift=0 # ignored at the moment
        if coarseFramesPerBin is not None:
            
            assert len(coarseSigma) == positions.shape[1]
            
            # print(f"Computing initial coarse drift estimate... ({coarseFramesPerBin} frames/bin)",flush=True, end='')
            with tqdm.tqdm(disable=True) as pbar:
                def update_pbar(i,info,drift_est): 
                    pbar.set_description(info); pbar.update(1)
                    if traces is not None:
                        traces.append(drift_est.copy())
                    return 1
    
                initial_drift,score = dll.MinEntropyDriftEstimate(
                    positions, framenum, initial_drift*1, coarseSigma, coarseMaxIterations, step, maxdrift, 
                    framesPerBin=coarseFramesPerBin, cuda=useCuda,progcb=update_pbar)
                
        with tqdm.tqdm(disable=True) as pbar:
            def update_pbar(i,info,drift_est): 
                pbar.set_description(info);pbar.update(1)
                if traces is not None:
                    traces.append(drift_est.copy())
                return 1
            update_pbar(0,'',initial_drift*1)
            drift,score = dll.MinEntropyDriftEstimate(
                positions, framenum, initial_drift*1, crlb, maxIterations, step, maxdrift, framesPerBin=framesperbin, maxneighbors=maxNeighbors,
                cuda=useCuda, progcb=update_pbar)
                
    
        drift -= np.mean(drift,0)                  
        return drift



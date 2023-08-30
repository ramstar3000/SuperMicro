import PySimpleGUI as Psg


def get_directory():

    '''
    This function creates a GUI that allows the user to select the directory containing the images.

    Returns
    -------
    directory : str
        The directory containing the images.
    
    '''

    # Select a neutral theme
    Psg.theme('Light Blue 2')

    Psg.set_options(font=("Helvetica", 10))

    layout = [[Psg.Text("Choose the directory containing the images", font=("Helvetica", 12))],
              [Psg.Input(default_text="C:/Users/RamVi/Documents/full_stack", size=(30, 1),
                         expand_x=True, font=("Helvetica", 12)), Psg.FolderBrowse()],
              [Psg.Text("The images should be named in the following format:")],
              [Psg.Text("XnYnRnWnCn.tiff or XnYnRnWn.tiff")],
              [Psg.Text("where n is the coordinate of the image.")],
              [Psg.Text("For example: X1Y1R1W1C1.tiff")],
              [Psg.Text(
                  "Please make sure parameters.txt is in the same directory as the images.")],
                [Psg.Text("""Optional: If you want sorting please provide the plate.xlsx file in the same directory, 
                          this must have a sheet called locations.""")],            
              [Psg.Button("Help with parameters.txt")],
              [Psg.Multiline(" ", key="Help", visible=False, size=(50, 20))],
              [Psg.OK()],]

    window = Psg.Window(
        "Choose the directory containing the images", layout, element_justification='c')

    while True:

        event, values = window.read()

        if event == Psg.WIN_CLOSED:

            raise SystemExit("No directory selected")

        if event == "OK":
            directory = values[0]
            break

        if event == "Help with parameters.txt":
            window['Help'].update(visible=True)
            window['Help'].update(
                """Here is an example of what parameters.txt should look like: \n

                \n {"estimate_fits_image":600, "tolerance":0.0001, "sorting":false,
  "max_iterations":20, "start":500, "stop":0,
  "drift":"RCC",  "timebin":10, "zoom":2,"num_beads":2, "drift_graphs":true, "bead_dist":200
  "CameraPixelSize":100, "roiRadius":5,
  "cameraOffset": 100, "cameraEMGain":1, "cameraQE":0.95, "cameraConversionFactor":1,
 "filenum":27, "frames":4000, "max_sigma":200, "max_localisation_precision":20, 
"grouping_distance":50, "thresholding_sigmas":4 } \n
                
                """)

    window.close()

    return directory


def main_menu():

    '''
    This function creates a GUI where the user can read the output of the program.    
    '''

    Psg.theme('Light Blue 2')

    Psg.set_options(font=("Helvetica", 10))

    layout = [
        [Psg.Multiline(key="output", size=(50, 10),
                       disabled=True, autoscroll=True)],
        [Psg.Cancel(button_color=('black', "#E3242B"))],
    ]

    window = Psg.Window("Main Menu", layout,
                        element_justification='c', size=(500, 300), finalize=True, resizable=True)

    return window

def error_message(message, n):

    # Displays message in a pop up window

    Psg.theme('Light Blue 2')

    Psg.set_options(font=("Helvetica", 10))
    
    layout = [
        [Psg.Text(message)],
        [Psg.OK(button_color=('black', "#E3242B"))],
    ]


    if n == 1:

        layout = [
            [Psg.Text(message)],
            [Psg.Text("The parameters file seems to have a problem. Please check the parameters file and try again.")],
            [Psg.OK(button_color=('black', "#E3242B"))],
         ]
        

    
    window = Psg.Window("Error", layout,
                        element_justification='c', size=(250, 300), finalize=True, resizable=True)
    
    while True:
            
            event, values = window.read()
    
            if event == Psg.WIN_CLOSED or event == "OK":
                break

    window.close()
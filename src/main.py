# -*- coding: utf-8 -*-
"""
########   #######   ######   ######   #######  
##     ## ##     ## ##    ## ##    ## ##     ## 
##     ## ##     ## ##       ##       ##     ## 
########  ##     ##  ######  ##       ##     ## 
##   ##   ##     ##       ## ##       ##     ## 
##    ##  ##     ## ##    ## ##    ## ##     ## 
##     ##  #######   ######   ######   #######  
------------------------------------------------
All rights reserved by Rosco Vision Systems.

Created on Wed Sep  7 17:18:43 2022
@author: Cody Hager
"""

# =============================================================================
# imports
# =============================================================================
import logging
import os
import PySimpleGUI as sg
import imageio.v3 as iio
import numpy as np
from modules.imgproc import ImageProcessor
# =============================================================================
# Globals
# =============================================================================
APP_VER = 'v0.1.0'
BTN_ENABLE_COLOR = None
BTN_NEUTRAL_COLOR = None
SCREEN_W, SCREEN_H = sg.Window.get_screen_size()
# =============================================================================
# window layouts
# =============================================================================

def main_win():
    width = int(SCREEN_W * 0.6)
    height = int(SCREEN_H * 0.6)
    
    menu_def = [['&File',['&Open Images']]]

    draw_tools = [
        [sg.Button('Draw Rectangle'),sg.Button('Clear Rectangle')]
    ]

    settings = [
        [sg.Text('Min Distance'),sg.Input('10',key='-MIN-DIST-')],
        [sg.Text('Window Size'),sg.Input('5',key='-WIN-SIZE-')],
        [sg.Text('Threshold'),sg.Slider((0.01,1.00),0.02,0.01,0.10,'h')]
    ]

    main_layout = [
        [sg.Menu(menu_def)],
        [
            sg.Graph(
             (width,height),
             graph_bottom_left=(0,0),
             graph_top_right=(width,height),
             enable_events=True,
             drag_submits=True,
             key='-GRAPH-'
            ),
            sg.Frame('AI Settings',layout=settings,expand_y=True)
        ],
        [sg.HorizontalSeparator()],
        [sg.Frame('Draw Tools',layout=draw_tools,expand_x=True)]   
    ]#end main_layout
    
    return sg.Window(f'Distortion Tester - {APP_VER}',layout=main_layout,resizable=True,finalize=True)

def img_browser_win():
    image_browser_layout = [
        [sg.Text('Images must be in PNG format')],
        [sg.Text('Image 1'),sg.Input(k='-IMG1 PATH-'),sg.FileBrowse(file_types=(('JPG','*.jpg'),('PNG','*.png')))],
        [sg.Button('Apply'),sg.Button('Process'),sg.Cancel()]
    ]
    
    return sg.Window('Image Browser',layout=image_browser_layout,finalize=True)
# =============================================================================
# helper methods
# =============================================================================
def logs_init():
    logs_dir = '../logs'
    if not os.path.isdir(logs_dir):
        os.mkdir(logs_dir)
    logging.basicConfig(filename=f'{logs_dir}/app.log',filemode='w',level=logging.DEBUG)

def get_img_bs(img) -> tuple:
    img = iio.imread(img)
    data = iio.imwrite('<bytes>',img,extension='.png')
    height = np.size(img,0)
    return height, data
  
# =============================================================================
# main 
# =============================================================================
if __name__ == '__main__':
    logs_init()
    logging.debug('entered main')
    ip = ImageProcessor()
    
    #initial state vars
    draw_rect = False   #flag for Draw Rectangle btn
    rect = {
        'top_left': None,
        'bottom_right': None,
        'fill_color': '',
        'line_color': 'red',
        'line_width': 2,
        'rect_id': None
        }
    
    #initialize windows
    sg.theme('BrownBlue')
    main_window = main_win()
    img_browser_window = None
    logging.debug('windows init')
    
    
    
    logging.debug('begin main loop')
    while True:
        window ,events, values = sg.read_all_windows()
        print(f'Events: {events}\nValues: {values}')
        
        if window == sg.WIN_CLOSED:
            logging.debug('program exiting')
            break
        
        if events == sg.WIN_CLOSED or events == None or events == 'Cancel':
            window.close()
            
            if window == main_window:
                main_window = None
            
            elif window == img_browser_window:
                img_browser_window = None
        
        #main window
        elif window == main_window:
            graph = main_window['-GRAPH-']
            draw_btn = main_window['Draw Rectangle']
            height,data = None,None
                
            if events == 'Open Images':
                img_browser_window = img_browser_win()
                
            elif events == 'Draw Rectangle':
                if not draw_rect:
                    draw_btn.update(button_color=BTN_ENABLE_COLOR)
                    draw_rect = True
                else:
                    draw_btn.update(button_color=BTN_NEUTRAL_COLOR)
                    draw_rect = False
            
            #Get corners of rectangle
            elif events == '-GRAPH-' and draw_rect and values['-GRAPH-'] != None and rect['top_left'] == None:
                rect['top_left'] = values['-GRAPH-']
            
            elif events == '-GRAPH-+UP' and draw_rect and rect['bottom_right'] == None:
                rect['bottom_right'] = values['-GRAPH-']
                top_left, bottom_right, fill_color, line_color, line_width, rect_id = [rect[k] for k in rect]
                
                rect['rect_id'] = graph.draw_rectangle(top_left,bottom_right,fill_color,line_color,line_width)
                
                #return button back to original state to indicate drawing is done
                draw_btn.update(button_color=BTN_NEUTRAL_COLOR)
                draw_rect = False
            
            elif events == 'Clear Rectangle' and rect['rect_id'] != None:
                graph.delete_figure(rect['rect_id'])
                rect['rect_id'] = None
                rect['top_left'] = None
                rect['bottom_right'] = None
        
        #img browser window
        elif window == img_browser_window:
            if events == 'Apply':
                height, data = get_img_bs(values['-IMG1 PATH-'])

                main_window['-GRAPH-'].update()
                main_window['-GRAPH-'].draw_image(data=data,location=(0,height*2))
                window.close()
                img_browser_window = None
            elif events == 'Process':
                ip.set_img(values['-IMG1 PATH-'])
                corners = ip.detect_corners()
                center_pnt = ip.get_center_point(corners)
                sorted_pnts = ip.get_points_distance_from_center(corners,center_pnt)
                grouped_pnts = ip.get_grouped_points(sorted_pnts,8)
                ip.draw_points(grouped_pnts,center_pnt)
                
    logging.debug('program stop')
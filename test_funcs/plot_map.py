import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from colour import Color
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import os

# Converting Shapefile Data Into Pandas Dataframes:
def read_shapefile(sf):
    #fetching the headings from the shape file
    fields = [x[0] for x in sf.fields][1:]
    #fetching the records from the shape file
    records = [list(i) for i in sf.records()]
    shps = [s.points for s in sf.shapes()]
    #converting shapefile data into pandas dataframe
    df = pd.DataFrame(columns=fields, data=records)
    #assigning the coordinates
    df = df.assign(coords=shps)
    return df


def plot_map(df, text = True, x_lim = None, y_lim = None, id_list = None, figsize = (11,9)):
    plt.figure(figsize = figsize)
    id=0
    for coord in df.coords:
        # plot points
        x = [i[0] for i in coord]
        y = [i[1] for i in coord]
        plt.plot(x, y, 'k')
        # 
        if text & (x_lim == None) & (y_lim == None):
            x0 = np.mean(x)
            y0 = np.mean(y)
            plt.text(x0, y0, id, fontsize=10)
        id = id+1
    
    if (x_lim != None) & (y_lim != None):     
        plt.xlim(x_lim)
        plt.ylim(y_lim)

def calc_color(data, color, num_color):
        if color   == 1: 
            color_sq =  ['#dadaebFF','#bcbddcF0','#9e9ac8F0','#807dbaF0','#6a51a3F0','#54278fF0']; 
            color_sq = list(Color("#dadaeb").range_to(Color("#6a51a3"),num_color))
            color_sq = [c.hex for c in color_sq]
            colors = 'Purples';
        elif color == 2: 
            color_sq = ['#effcef','#94d3ac','#50d890','#55ae95']; 
            colors = 'Greens';
            color_sq = list(Color(color_sq[0]).range_to(Color(color_sq[-1]),num_color))
            color_sq = [c.hex for c in color_sq]
        elif color == 3: 
            color_sq = ['#f7f7f7','#d9d9d9','#bdbdbd','#969696','#636363','#252525']; 
            colors = 'Greys';
            color_sq = list(Color(color_sq[0]).range_to(Color(color_sq[-1]),num_color))
            color_sq = [c.hex for c in color_sq]
        elif color == 4: 
            color_sq = ['#f8fbff','#eaf4ff', '#d6eaff','#add6ff','#84c1ff']; 
            colors = 'Blues';
            color_sq = list(Color(color_sq[0]).range_to(Color(color_sq[-1]),num_color))
            color_sq = [c.hex for c in color_sq]
        elif color == 9: 
            color_sq = ['#ff0000','#ff0000','#ff0000','#ff0000','#ff0000','#ff0000'];
            color_sq = list(Color(color_sq[0]).range_to(Color(color_sq[-1]),num_color))
            color_sq = [c.hex for c in color_sq]
        else:           
            color_sq = ['#ffffd4','#fee391','#fec44f','#fe9929','#d95f0e','#993404']; 
            colors = 'YlOrBr';
            num_color = 15
            color_sq = list(Color(color_sq[0]).range_to(Color(color_sq[-1]),num_color))
            color_sq = [c.hex for c in color_sq]

        new_data, bins = pd.qcut(data, num_color, retbins=True, 
        labels=list(range(num_color)))
        color_ton = []
        for val in new_data:
            color_ton.append(color_sq[val]) 
        if color != 9:
            colors = sns.color_palette(colors, n_colors=num_color)
            sns.palplot(colors, 0.6);  
        return color_ton, bins, color_sq

def plot_cities_data(df, title, ids, data=None,color=None, num_color=15, print_id=False):
    color_ton, bins, color_sq = calc_color(data, color, num_color)
    ax = plot_map_fill_multiples_ids_tone(df, title, ids, 
                                     print_id, 
                                     color_ton, 
                                     bins, data,
                                     color_sq,
                                     x_lim = None, 
                                     y_lim = None);
    return ax

def plot_map_fill_multiples_ids_tone(df, title, ids,  
                                     print_id, color_ton, 
                                     bins, data = None,
                                     color_sq = None,
                                     x_lim = None, 
                                     y_lim = None, 
                                     show_title = False,
                                     figsize = (12,9)):
   
        
    plt.figure(figsize = figsize)
    fig, ax = plt.subplots(figsize = figsize)
    if show_title:
        fig.suptitle(title, fontsize=16)
    for coord in df.coords:
        x = [i[0] for i in coord]
        y = [i[1] for i in coord]
        ax.plot(x, y, 'k')
            
    for id in ids:
        #shape_ex = sf.shape(id)
        points = df.coords[id]
        x_lon = np.zeros((len(points),1))
        y_lat = np.zeros((len(points),1))
        for ip in range(len(points)):
            x_lon[ip], y_lat[ip] = points[ip]
        ax.fill(x_lon,y_lat, color_ton[id])
        if print_id != False:
            x0 = np.mean(x_lon)
            y0 = np.mean(y_lat)
            plt.text(x0, y0, id, fontsize=10)
    if (x_lim != None) & (y_lim != None):     
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    # cax, _ = cbar.make_axes(ax) 
    cax = fig.add_axes([0.45, 0.2, 0.25, 0.03])
    normal = plt.Normalize(data.min(), data.max())
    cb2 = cbar.ColorbarBase(cax, cmap=ListedColormap(color_sq),norm=normal, orientation='horizontal') 
    return ax
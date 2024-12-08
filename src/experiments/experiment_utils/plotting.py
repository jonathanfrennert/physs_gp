page_width_cm = 17.14256
column_width_cm = 8.25381
cm_to_inch =  0.393701

_scriptsize = 7
_linewidth = 1.5
_dpi=300

settings = {
    'page_width_cm': page_width_cm, 
    'column_width_cm': column_width_cm, 
    'page_width_inch': page_width_cm*cm_to_inch, 
    'column_width_inch': column_width_cm*cm_to_inch, 
    'font_sizes': {
        'ticks': _scriptsize, 
        'legend': _scriptsize, 
        'label': _scriptsize 
    },
    'linewidth': _linewidth,
}

#====================== Default configs ======================
color_palette = {
    'black': '#000000',
    'orange': '#E69F00',
    'blue': '#56B4E9',
    'green': '#009E73',
    'orange': '#F0E442',
    'dark_blue': '#0072B2',
    'dark_orange': '#D55E00',
    'pink': '#CC79A7',
    'white': '#111111',
    'grey': 'grey'
}

#====================== Model Specifics configs ======================

model_configs = {
    'batch': {
        'name': r'\gp',
        'color': color_palette['black'],
        'linestyle': '-'
    },
    'sde': {
        'name': r'\PIGP',
        'color': color_palette['green'],
        'linestyle': '-'
    },
    'sparse_sde': {
        'name': r'\sPIGP',
        'color': color_palette['green'],
        'linestyle': '--'
    },
    'hierarchical_sde': {
        'name': r'\hPIGP',
        'color': color_palette['green'],
        'linestyle': 'dotted'
    },
    'sparse_hierarchical_sde': {
        'name': r'\hsPIGP',
        'color': color_palette['green'],
        'linestyle': 'dashdot'
    },
    'autoip': {
        'name': r'\PIGP',
        'color': color_palette['dark_orange'],
        'linestyle': '-'
    },
}


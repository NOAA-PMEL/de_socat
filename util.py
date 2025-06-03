from sdig.erddap.info import Info

import urllib

def make_con(erddap_variable_name, menu_input):   
    con = ''
    if menu_input is not None and len(menu_input) > 0:
        con_dict = Info.make_platform_constraint(erddap_variable_name, menu_input)
        print(con_dict['con'])
        con = '&'+con_dict['con']
    return con
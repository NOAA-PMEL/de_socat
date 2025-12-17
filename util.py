from numpy import isin
from sdig.erddap.info import Info

import urllib

def make_con(erddap_variable_name, menu_input):   
    con = ''
    if menu_input is not None and len(menu_input) > 0:
        con_dict = Info.make_platform_constraint(erddap_variable_name, menu_input)
        # DEBUG print(con_dict['con'])
        con = '&'+con_dict['con']
    return con

def make_query(column_name, menu_input, add_and):
    con = ''
    if menu_input is not None and len(menu_input) > 0:
        if isinstance(menu_input, list):
            con = "','".join(menu_input)
            con = "'" + con + "'"
            con = f'{column_name} IN ({con})'
        elif isinstance(menu_input, str):
            con = f"{column_name} = '{menu_input}'"
        else:
            con = f"{column_name} = {menu_input}"
        if add_and:
            con = ' AND ' + con
    return con
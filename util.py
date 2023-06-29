from sdig.erddap.info import Info

def make_con(erddap_variable_name, menu_input):   
    con = ''
    if menu_input is not None and len(menu_input) > 0:
        con_dict = Info.make_platform_constraint(erddap_variable_name, menu_input)
        con = '&' + con_dict['con']
    return con
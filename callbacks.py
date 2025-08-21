from dash import Input, Output, State, no_update
import constants
import json
from io import StringIO
import constants
import pandas as pd
from datetime import date, datetime, timezone
redis_instance = constants.redis_instance
import db




def register_editor_callbacks(app):

    @app.callback(
        [
            Output('add-qc-dialog', 'expanded', allow_duplicate=True),
        ],
        [
            Input('cruise-qc-cancel', 'n_clicks'),
        ], prevent_initial_call=True
    )
    def close_add_qc(click):
        return [False]

    @app.callback(
        Output('save-full-message', 'children'),
        Output('save-full-message-card', 'style', allow_duplicate=True),
        Output('add-qc-dialog', 'expanded', allow_duplicate=True),
        Input('cruise-qc-save', 'n_clicks'),
        State('qc-region-value', 'value'),
        State('added-qc-flag', 'value'),
        State('fco2-comment', 'value'),
        State('sop-comment', 'value'),
        State('meta-comment', 'value'),
        State('data-comment', 'value'),
        State('xover-comment', 'value'),
        State('qc-additional-comment', 'value'),
        State('plot-expocode', 'value'),
        prevent_initial_call=True
    ) 
    def show_and_save_comments(click, qc_region_value, flag_value, fco2, sop, meta, data, xover, additional_comment, expocode):
        print('qc flag saved')
        full_comment = ''
        if fco2 is not None and fco2 != 'fco2no':
            full_comment = full_comment + socatQC[fco2]
            print('added a comment ', full_comment)
        if sop is not None and sop != 'sopno':
            if len(full_comment) > 0:
                full_comment = full_comment + socatQC['commentSpacer']
            full_comment = full_comment + socatQC[sop]
            print('add sop comment ', full_comment)
        if meta is not None and meta != 'metano':
            if len(full_comment) > 0:
                full_comment = full_comment + socatQC['commentSpacer']
            full_comment = full_comment + socatQC[meta]
        if data is not None and data != 'datano':
            if len(full_comment) > 0:
                full_comment = full_comment + socatQC['commentSpacer']
            full_comment = full_comment + socatQC[data]
        if xover is not None and xover != 'crossno':
            if len(full_comment) > 0:
                full_comment = full_comment + socatQC['commentSpacer']
            full_comment = full_comment + socatQC[xover]
        if len(full_comment) > 0:
            full_comment = full_comment + '.'
        if len(additional_comment) > 0:
            full_comment = full_comment + ' ' + additional_comment
        print ('final comment ', full_comment)
        df_json_string = redis_instance.hget('cache', 'table-of-cruises').decode('utf-8')
        df = pd.read_json(StringIO(json.loads(df_json_string)))
        row = df.loc[df['expocode']==expocode]
        socat_version = str(row['socat_version'])
        # TODO we need to know who is logged in
        # Actually save the stuff instead of returning a query string
        frame_input = {'qc_flag': flag_value, 'qc_time': datetime.now(timezone.utc).isoformat(), 'expocode': expocode, 'socat_version': socat_version, 'region_id': qc_region_value, 'reviewer_id': 'FAKE REVIEWER', 'qc_comment': full_comment}
        df = pd.DataFrame(frame_input)
        df.to_sql(constants.qc_entries_table, postgres_engine, if_exists='append', index=False)
        return [full_comment, {'visibility':'visible'}, False]

    @app.callback(
        [
            Output('selected-points', 'rowData', allow_duplicate=True),
            Output('selected-points', 'selectedRows')
        ],
        [
            Input('set-woce-flag', 'n_clicks')
        ],
        [
            State('selected-points', 'rowData'),
            State('selected-points', 'selectedRows'),
            State('qc-woce-co2-flag-value', 'value'),
            State('woce-flag-to-set', 'value')
        ], prevent_initial_call = True
    )
    def apply_woce_water(click, row_data, selected_rows, woce_flag, flag_to_set):
        if row_data is None or len(row_data) < 1:
            return no_update
        if woce_flag is None or len(woce_flag) < 1:
            return no_update
        if selected_rows is None or len(selected_rows) < 1:
            return no_update
        for srow in selected_rows:
            time = srow['time']
            for row in row_data:
                if row['time'] == time:
                    row[flag_to_set] = int(woce_flag)
        return row_data, []
        

    @app.callback(
        [
            Output('comment', 'value'),
            Output('save-full-message-card', 'style', allow_duplicate=True),
            Output('save-full-message', 'children', allow_duplicate=True),
            Output('edit-flags-modal', 'expanded')
        ],
        [
            Input('save-woce-flags', 'n_clicks')
        ],
        [
            State('selected-points', 'rowData'),
            State('comment','value'),
        ], prevent_initial_call=True
    )
    def save_flags(click, edited_points, in_comment):
        reminder = 'You must supply a comment.'
        ex_reminder = 'No, really. You must supply a comment telling what you did and why.'
        if in_comment is None or len(in_comment) == 0 or in_comment == reminder or in_comment == ex_reminder:
            if in_comment == reminder:
                return ex_reminder, no_update, no_update, True
            else:
                return reminder, no_update, no_update, True
        selected_data_string = redis_instance.hget("cache","edit-table-data").decode('utf-8')
        selected_data_json = json.loads(selected_data_string)
        selected_data = pd.read_json(StringIO(selected_data_json))
        as_edited = pd.DataFrame(edited_points)
        edits = pd.concat([selected_data, as_edited]).drop_duplicates(keep=False)
        start = int(edits.shape[0]/2)
        d = datetime.now(timezone.utc)
        d = str(d)
        d = d.replace(d[-7:], 'Z')
        edits.loc[:, 'edit_timestamp'] = d
        edits.loc[:, 'comment'] = in_comment
        save_edits = edits.iloc[start:]
        save_edits.to_sql(constants.edits_table, postgres_engine, if_exists='append', index=False)
        # save_edits.to_sql(edits_table, db.mysql_engine, if_exists='append', index=False)
        return '', {'visibility':'visible'}, f'The WOCE flag changes have been saved.  {save_edits.shape[0]} rows changed.', False


    @app.callback(
        [
            Output('edited-points', 'rowData'),
            Output('edited-points', 'columnDefs'),
        ],
        [
            Input('show-edits', 'n_clicks')
        ], prevent_initial_call=True
    )
    def show_edits(clicked):
        edited_rows = db.show_saves()
        columnDefs=[{"field": i, "headerName": i} for i in sorted(edited_rows.columns, key=str.casefold)]
        return [edited_rows.to_dict("records"), columnDefs]



    @app.callback(
        [
            Output('qc-entries', 'rowData'),
            Output('qc-entries', 'columnDefs')
        ],
        [
            Input('show-qc-entries', 'n_clicks')
        ]
    )
    def show_qc(clicked):
        new_qc = db.show_qc()
        columnDefs=[{"field": i, "headerName": i} for i in sorted(new_qc.columns, key=str.casefold)]
        return [new_qc.to_dict("records"), columnDefs]


    @app.callback(
        [
            Output('selected-points', 'rowData'),
            Output('selected-points', 'columnDefs'),
            Output('selected-points',  'rowClassRules'),
            Output('set-row-item', 'label')
        ],
        [
            Input('flag', 'n_clicks')
        ],
        [
            State('prop-prop-graph', 'selectedData'),
            State('woce-flag-to-set', 'value')
        ]
    )
    def show_selected_points(click, in_points, flag_to_set):
        all_data_string = redis_instance.hget("cache","plot-data").decode('utf-8')
        all_data_json = json.loads(all_data_string)
        all_data = pd.read_json(StringIO(all_data_json))
        if flag_to_set == 'WOCE_CO2_atm':
            row_rules = constants.atm_edit_style
        else:
            row_rules = constants.water_edit_style
        column_names = sorted(all_data.columns, key=str.casefold)
        column_names.remove('WOCE_CO2_water')
        column_names.remove('WOCE_CO2_atm')
        column_names.insert(0, flag_to_set) 
        row_item_label = f'Set {flag_to_set} Checked Rows:'
        if in_points is not None:
            # TODO These are the columns from the plot, maybe we should use the columns defined as necessary for setting the flags
        
            columnDefs = []
            for idx, i in enumerate(column_names):
                if 'time' in i:
                    columnDefs.append({"field": i, "headerName": i, 'sortable': True})
                elif 'WOCE' in i:
                    if idx == 0:
                        checks = True
                    else:
                        checks = False
                    columnDefs.append({
                        "field": i, "headerName": i, 
                        "checkboxSelection": checks,
                        'cellEditorParams': {'values': [2, 3, 4]},
                        "editable": True,
                        'cellEditor': 'agSelectCellEditor',
                        'cellClassRules': {
                            'green-cell': 'params.value == 2',
                            'yellow-cell': 'params.value == 3',
                            'red-cell': 'params.value == 4',
                        },
                        'sortable': True
                    })
                else:
                    columnDefs.append({"field": i, "headerName": i, 'sortable': True})
            selected_points = in_points['points']
            times = []
            for point in selected_points:
                customs = point['customdata']
                times.append(customs[0])
            to_show = all_data.loc[all_data['time'].isin(times)]
            redis_instance.hset("cache", 'edit-table-data', json.dumps(to_show.to_json()))
            return [to_show.to_dict("records"), columnDefs, row_rules, row_item_label]
        else:
            to_show = pd.DataFrame(columns=column_names)
            return [to_show.to_dict("records"), no_update, no_update, 'No points where selected. Use the Box Select tool to select points to edit.']

var dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};
dagcomponentfuncs.CustomTooltip = function (props) {
    info = [
        React.createElement('h5', {}, 'Double Click to Change Flag'),
    ];
    return React.createElement(
        'div',
        {
            style: {
                border: '2pt solid black',
                backgroundColor: 'white',
                padding: 3,
            },
        },
        info
    );
};
dagcomponentfuncs.DocLink = function (props) {
    return React.createElement('a',
    {
        target: '_blank',
        href: props.value
    }, 'Open!');
};

// buttonCellRenderer.js
// This JavaScript function will be used as a cell renderer in Dash AG Grid.
// It takes the cell's parameters (params) as input and returns an HTML element
// containing four styled buttons.

dagcomponentfuncs.myButtonCellRenderer = function(props) {
        const {setData, data} = props;
    
        function onClick() {
            setData();
        }
        const myStyles = {
            height: '30px',
            marginTop: '3px',
            marginBottom: '6px'
        };
        return React.createElement(
            'button',
            {
                onClick: onClick,
                className: props.className,
                style: myStyles
            },
            'Go!'
        );
    };

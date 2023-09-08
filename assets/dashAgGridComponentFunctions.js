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
    }, 'Documentation');
};
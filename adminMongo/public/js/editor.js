$(document).ready(function(){
    if (document.getElementById('json')) {
        var editor = ace.edit('json');
        editor.setTheme('ace/theme/github');
        editor.session.setMode('ace/mode/python');
        editor.setFontSize(14);
        editor.getSession().setUseWorker(false);
        editor.$blockScrolling = Infinity;
        $(document).on('click', '#submit_json', function(){
            try{
                // convert BSON string to EJSON
                var ejson = toEJSON.serializeString(editor.getValue());
                $.ajax({
                    method: 'POST',
                    contentType: 'application/json',
                    url: $('#app_context').val() + '/document/' + $('#conn_name').val() + '/' + $('#db_name').val() + '/' + $('#coll_name').val() + '/' + $('#edit_request_type').val(),
                    data: JSON.stringify({'objectData': ejson})
                })
                .done(function(data){
                    show_notification(data.msg, 'success');
                    if(data.doc_id){
                        setInterval(function(){
                            // remove "new" and replace with "edit" and redirect to edit the doc
                            window.location = window.location.href.substring(0, window.location.href.length - 3) + 'edit/' + data.doc_id;
                        }, 2500);
                    }
                })
                .fail(function(data){
                    show_notification(data.responseJSON.msg, 'danger');
                });
            }catch(err){
                show_notification(err, 'danger');
            }
        });
    }
    if (document.getElementById('diff_editor')) {
        var diff_srcs = JSON.parse(document.getElementById('diff_editor').dataset.raw);
        var diffEditor = new AceDiff({
            element: '#diff_editor',
            mode: 'ace/mode/python',
            left: {
                content: diff_srcs['original_code']
            },
            right: {
                content: diff_srcs['mutated_code']
            },
        });
        // set attributes for both left and right editors
        for (const [_, editor] of Object.entries(diffEditor.getEditors())) {
            if (editor) {
                editor.setFontSize(14);
                editor.getSession().setUseWorker(false);
                editor.$blockScrolling = Infinity;
            }
        };
    }  
});

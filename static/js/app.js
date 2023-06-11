var entity
var html
var sentence
var i
var start
var end

jQuery(document).ready(function () {
    var slider_sentences = $('#max_len')

    $(document).ready(function () {
        var slider_sentences = $('#max_len')
    
        $(document).on('click', '#btn_generate', function (e) {
            if ($('#input_text').val() == "") {
                alert('Insert a text');
                return;
            }
            $.ajax({
                url: '/process',
                type: "post",
                contentType: "application/json",
                dataType: "json",
                data: JSON.stringify({
                    "input_text": $('#input_text').val(),
                    "return_probability": $('#return_probability').val()
                }),
                beforeSend: function () {
                    $('.overlay').show();
                    $('#result').html('');
                },
                complete: function () {
                    $('.overlay').hide();
                }
            }).done(function (jsondata, textStatus, jqXHR) {
                // Update the result container with the received HTML
                $('#result').html(jsondata.html);
    
                // Uncomment the following lines if you want to display the JSON data as well
                // var jsonDataElement = $('<pre>').text(JSON.stringify(jsondata, null, 2));
                // $('#result').append(jsonDataElement);
    
            }).fail(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata);
            });
        });
    });

})
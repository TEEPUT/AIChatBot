//===============================================
// AI 채팅 박스
//===============================================
function typeWriter(element, text, i, fnCallback) {
    if (i < text.length) {
        if (text.charAt(i) === '<') {
            var tag = '';
            while (text.charAt(i) !== '>') {
                tag += text.charAt(i);
                i++;
            }
            tag += '>';
            element.innerHTML += tag;
            i++;
        } else {
            element.innerHTML += text.charAt(i);
            i++;
        }
        setTimeout(function() {
            typeWriter(element, text, i, fnCallback);
        }, 10);
    } else if (typeof fnCallback == 'function') {
        setTimeout(fnCallback, 700);
    }
}


function scrollToBottom() {
    $('#chatbox').animate({scrollTop: $('#chatbox').prop('scrollHeight')}, 700);
}

   
function animateBotResponse(response) {
    var botMessage = $('<div class="bot-message"></div>');
    $('#chatbox').append(botMessage);
    typeWriter(botMessage[0], response, 0, function() {
        scrollToBottom();
    });
}

$(function() {
    $('#input-form').submit(function(event) {
        event.preventDefault();
        var user_input = $('#input_data').val();
        if (!user_input) return;
        $('#chatbox').append('<div class="user-message">' + user_input + '</div>');
        $('#input_data').val('');
        scrollToBottom();
        $.ajax({
            url: '/get_answer',
            type: 'POST',
            data: JSON.stringify({question: user_input}),
            contentType: 'application/json',
            success: function(data) {
                var bot_response = data.answer;
                bot_response = bot_response.replace(/\n/g, '<br>');
                animateBotResponse(bot_response);
            }
        });
    });
});

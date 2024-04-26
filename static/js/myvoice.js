//===============================================
// 음성 인식 및 변환
//===============================================
var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
var recognition = new SpeechRecognition();
recognition.lang = 'ko-KR'; // 한국어로 설정
recognition.interimResults = true; // 중간 결과를 반환
recognition.continuous = true; // 연속적인 결과를 반환

var recognitionTimeout; // 음성이 없는 경우를 확인하기 위한 타이머 변수

recognition.onstart = function() {
    console.log('Voice recognition started. Try speaking into the microphone.');
    $('#start-btn').addClass('action'); // 음성 인식이 시작되면 class="action" 추가
}

recognition.onend = function() {
    $('#start-btn').removeClass('action');
    $('#user-input').val(''); // 음성 인식 종료 후 인풋 필드 리셋
}

recognition.onend = function() {
    $('#start-btn').removeClass('action'); // 음성 인식이 종료되면 class="action" 제거
}
recognition.onresult = function(event) {
    var transcript = event.results[event.results.length - 1][0].transcript;
    document.getElementById('input_data').value = transcript;
    clearTimeout(recognitionTimeout); // 결과가 도착하면 타이머를 제거

     recognitionTimeout = setTimeout(() => { // 3초 후에 음성 인식을 중단
     recognition.stop();
    }, 3000);
}

document.querySelector('#start-btn').addEventListener('click', function(e) {
    recognition.start();
});
 
// 전송 버튼이 클릭되면 음성 인식이 중단되도록 합니다.
document.querySelector('#input-form button[type=submit]').addEventListener('click', function(e) {
    recognition.stop();
});
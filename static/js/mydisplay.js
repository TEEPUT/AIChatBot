//===============================================
// 화면 사이즈 측정
// DelftStack에서 가져옴
//===============================================
function return_screen_size() {
    // Get the size of the device screen
    var screenWidth = screen.width;
    var screenHeight = screen.height;
    // Get the browser window size
    var windowWidth = window.innerWidth;
    var windowHeight = window.innerHeight;
    // Get the size of the entire webpage
    const scrollWidth = document.documentElement.scrollWidth;
    const scrollHeight = document.documentElement.scrollHeight;
    var str = "Device W:" + screenWidth + ", H:" + screenHeight + ".";
    str += "Browser W: " + windowWidth + ", H:" + windowHeight + ".";
    str += "Scroll W:" + scrollWidth + ", H:" + scrollHeight + ".";
    return str;
    }
    
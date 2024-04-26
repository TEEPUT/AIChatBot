//================================
// nodejs Main
//================================

var fs = require('fs');
var express = require('express');
var app = express();

//================================
app.use(express.static("static"));
app.use(express.json()); //json 형태로 parsing
app.use(express.urlencoded({ extended: false }));

//================================
app.get('/', function (req, res) {
    var url = '';
    if (req.url == '/favicon.ico') {
        return res.writeHead(404);
    } else if (req.url == '/') {
        url = '/templates/ai_hello.html';
    } else {
        url = req.url;
    }
    res.writeHead(200);
    msg = `html의 위치는:${__dirname}${url}`;
    console.log(msg);
    res.end(fs.readFileSync(__dirname + url));
});

//================================
app.get('/html/:htmlname', function (req, res) {
    var url = '';
    var pagename = '/templates/' + req.params.htmlname;
    url = pagename;
    res.writeHead(200);
    msg = `html의 위치는:${__dirname}${url}`;
    console.log('html의 위치는 ' + __dirname + url);
    res.end(fs.readFileSync(__dirname + url));
});

//================================
app.post('/get_answer', function (req, res) {
    var htmlStr = '';
    var quest1 = req.body.question;
    if (quest1 == '') quest1 = "질문을 하셔요~";
    const { spawn } = require("child_process");
    const Python = spawn("python", ["myutil/call_ai_api.py", quest1]);
    Python.stdout.on("data", (data) => {
        htmlStr += data;
    });
    Python.stderr.on("data", (data) => {
        htmlStr += data;
    });
    Python.on("close", (code) => {
        var resData = {};
        resData.answer = htmlStr;
        res.json(resData);
    });
});

//================================
app.get('/read', function (req, res) {
    var htmlStr = '';

    const { spawn } = require("child_process");
    const Python = spawn("python", ["myutil/read_database.py"]);

    Python.stdout.on("data", (data) => {
    htmlStr += data;
    });

    Python.stderr.on("data", (data) => {
    htmlStr += data;
    });

    Python.on("close", (code) => {
        res.writeHead(200, { 'Content-Type': 'text/html;charset=UTF-8' });

        res.write(htmlStr);
        res.end();
    });
});


//================================
const port = 5555;
app.listen(port, () => {
    console.log('Server listening on port ', port);
});

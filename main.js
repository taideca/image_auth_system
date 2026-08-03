let src, gray, orb, matcher;
let kp_correct, des_correct;
let isReady = false;
let isCorrectImageLoaded = false;

const statusDiv = document.getElementById('status');
const video = document.getElementById('video');

// 1. OpenCV.jsの読み込み完了時に実行される関数
function onOpenCvReady() {
    statusDiv.innerText = "正解データを読み込み中...";
    
    // ORB検出器とマッチング器の初期化
    orb = new cv.ORB();
    matcher = new cv.BFMatcher(cv.NORM_HAMMING, true);
    isReady = true;

    // サーバー上の正解画像を自動ロード
    loadCorrectImage('img/template.jpg');
}

// 2. 固定の正解画像を読み込んで特徴点を抽出する
function loadCorrectImage(imagePath) {
    let imgElement = new Image();
    imgElement.src = imagePath;
    
    imgElement.onload = function() {
        try {
            let srcCorrect = cv.imread(imgElement);
            let grayCorrect = new cv.Mat();
            cv.cvtColor(srcCorrect, grayCorrect, cv.COLOR_RGBA2GRAY);
            
            kp_correct = new cv.KeyPointVector();
            des_correct = new cv.Mat();
            
            // 正解画像の特徴点を抽出してメモリに保存
            orb.detectAndCompute(grayCorrect, new cv.Mat(), kp_correct, des_correct);
            
            // 使用済みのテンポラリメモリを解放
            srcCorrect.delete(); 
            grayCorrect.delete();
            
            isCorrectImageLoaded = true;
            statusDiv.innerText = "カメラを起動中...";
            startCamera();
        } catch (err) {
            statusDiv.innerText = "正解画像の解析に失敗しました。";
            statusDiv.className = "fail";
            console.error(err);
        }
    };

    imgElement.onerror = function() {
        statusDiv.innerText = "エラー: img/correct.jpg が見つかりません。";
        statusDiv.className = "fail";
    };
}

// 3. スマホの背面カメラを起動
function startCamera() {
    navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment" }, // 背面カメラを指定
        audio: false 
    })
    .then((stream) => {
        video.srcObject = stream;
        video.addEventListener('canplay', () => {
            // カメラ映像のサイズに合わせて処理用Mat（行列）を初期化
            src = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC4);
            gray = new cv.Mat();
            statusDiv.innerText = "判定中...";
            statusDiv.className = "loading";
            requestAnimationFrame(processVideo);
        });
    })
    .catch((err) => {
        statusDiv.innerText = "カメラの起動に失敗しました。アクセスを許可してください。";
        statusDiv.className = "fail";
        console.error(err);
    });
}

// 4. 毎フレームのリアルタイム画像処理ループ
function processVideo() {
    if (!isCorrectImageLoaded) return;

    let cap = new cv.VideoCapture(video);
    cap.read(src); // 現在のカメラフレームをsrcに読み込み
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    let kp_captured = new cv.KeyPointVector();
    let des_captured = new cv.Mat();
    orb.detectAndCompute(gray, new cv.Mat(), kp_captured, des_captured);

    let isMatch = false;
    let matchCount = 0;

    // 特徴点のマッチング計算
    if (!des_captured.empty() && !des_correct.empty()) {
        let matches = new cv.DMatchVector();
        try {
            matcher.match(des_correct, des_captured, matches);
            
            // 良いマッチ（距離が近い＝形が似ている特徴点）をカウント
            for (let i = 0; i < matches.size(); i++) {
                if (matches.get(i).distance < 45) { // しきい値（判定の厳しさ：数字が小さいほど厳格）
                    matchCount++;
                }
            }
            // 15個以上の特徴点が一致したら「合格」
            if (matchCount >= 15) { // 必要マッチ数（数字が大きいほど厳格）
                isMatch = true;
            }
        } catch(e) {
            console.error(e);
        }
        matches.delete();
    }

    // 判定結果をUIに反映
    if (isMatch) {
        statusDiv.innerText = `🎉 一致 (PASS) [Matches: ${matchCount}]`;
        statusDiv.className = "pass";
    } else {
        statusDiv.innerText = `❌ 不一致 (FAIL) [Matches: ${matchCount}]`;
        statusDiv.className = "fail";
    }

    // 画面の描画更新（映像を表示）
    cv.imshow('outputCanvas', src);

    // ループ内でのメモリリークを防ぐため、毎フレームのMatを解放
    kp_captured.delete();
    des_captured.delete();

    // 次のフレームの処理を予約（ループ実行）
    requestAnimationFrame(processVideo);
}

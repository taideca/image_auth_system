const statusDiv = document.getElementById('status');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

let correctFeatures = null;
let isLockedPass = false;

// リセットボタンを自動生成
createResetButton();

// ★GitHub Pagesのパス（必ず「/あなたのリポジトリ名/」に書き換えてください）
const REPO_NAME = "/image_auth_system/"; 

// 1. サーバー上の正解画像をロード
const img = new Image();
img.src = REPO_NAME + 'template.png';

img.onload = function() {
    try {
        context.drawImage(img, 0, 0, 320, 240);
        const imageData = context.getImageData(0, 0, 320, 240);
        const gray = tracking.Image.grayscale(imageData.data, 320, 240);
        const corners = tracking.Fast.getCornerPoints(gray, 320, 240);
        
        // 特徴点を抽出して保存
        correctFeatures = tracking.Brief.getDescriptors(gray, 320, corners);
        
        statusDiv.innerText = "カメラを起動中...";
        startCamera();
    } catch(err) {
        statusDiv.innerText = "❌ 正解画像の解析（tracking.js）に失敗しました: " + err.message;
        statusDiv.className = "fail";
    }
};

img.onerror = function() {
    statusDiv.innerText = "❌ 画像が見つかりません: " + img.src;
    statusDiv.className = "fail";
};

// 2. カメラ起動とトラッキング開始
function startCamera() {
    statusDiv.innerText = "判定中...";
    statusDiv.className = "loading";

    // リアルタイムトラッカーの定義
    const ImageTracker = function() {
        ImageTracker.prototype.track = function(pixels, width, height) {
            if (isLockedPass || !correctFeatures) return;

            try {
                const gray = tracking.Image.grayscale(pixels, width, height);
                const corners = tracking.Fast.getCornerPoints(gray, width, height);
                const currentFeatures = tracking.Brief.getDescriptors(gray, width, corners);
                
                // 特徴点マッチングの計算
                const matches = tracking.Brief.reciprocalMatch(correctFeatures, currentFeatures);
                let matchCount = matches.length;
                
                // 20個以上一致でPASS判定（状況に合わせて数値を調整してください）
                if (matchCount >= 20) { 
                    isLockedPass = true;
                    if (navigator.vibrate) navigator.vibrate(200);

                    statusDiv.innerText = `🎉 一致 (PASS) [Matches: ${matchCount}]`;
                    statusDiv.className = "pass";
                    document.getElementById('resetBtn').style.display = "block";
                } else {
                    statusDiv.innerText = `❌ 不一致 (FAIL) [Matches: ${matchCount}]`;
                    statusDiv.className = "fail";
                }
            } catch(e) {
                statusDiv.innerText = "🚨 計算エラー: " + e.message;
                statusDiv.className = "fail";
            }
        };
    };
    tracking.inherits(ImageTracker, tracking.Tracker);

    const myTracker = new ImageTracker();
    
    // tracking.jsでWebカメラを起動して追跡開始
    tracking.track('#video', myTracker, { camera: true })
    .on('error', function(event) {
        statusDiv.innerText = "❌ カメラの追跡エラーが発生しました: " + event.message;
        statusDiv.className = "fail";
    });
}

// 3. リセットボタンの作成
function createResetButton() {
    const btn = document.createElement('button');
    btn.id = "resetBtn";
    btn.innerText = "🔄 判定をリセットして再開";
    btn.style.cssText = "display:none; margin: 15px auto; padding: 12px 24px; font-size: 1rem; font-weight: bold; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;";
    
    btn.onclick = function() {
        isLockedPass = false;
        btn.style.display = "none";
        statusDiv.innerText = "判定中...";
        statusDiv.className = "loading";
    };
    statusDiv.parentNode.insertBefore(btn, statusDiv.nextSibling);
}

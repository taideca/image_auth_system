let video, cap, src, dst, gray, edges, templateMat;
let isRunning = false;
let isMatched = false;

// 1. OpenCV.js読み込み完了時の処理（修正版）
function onOpenCvReady() {
  // OpenCVの内部システム(WebAssembly)の準備が完了するまで待つ
  cv['onRuntimeInitialized'] = () => {
    document.getElementById('status').innerText = 'カメラを起動中...';
    startCamera();
  };
}

// カメラを起動する処理を切り出し
function startCamera() {
  video = document.getElementById('videoInput');

  // スマホの背面カメラを優先して起動
  navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 640, height: 480 }, audio: false })
    .then(function(stream) {
      video.srcObject = stream;
      video.play();
      video.oncanplay = () => {
        initOpenCV();
      };
    })
    .catch(function(err) {
      document.getElementById('status').innerText = 'カメラのアクセスに失敗しました: ' + err;
    });
}

// 2. 変数の初期化とループ開始
function initOpenCV() {
  src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  gray = new cv.Mat();
  edges = new cv.Mat();
  cap = new cv.VideoCapture(video);

  // テンプレート画像（正解画像）を読み込み、グレースケール化
  let templateImgElement = document.getElementById('templateImage');
  templateMat = cv.imread(templateImgElement);
  cv.cvtColor(templateMat, templateMat, cv.COLOR_RGBA2GRAY);

  document.getElementById('status').innerText = '枠をカメラに映してください';
  isRunning = true;
  requestAnimationFrame(processVideo);
}

// 3. 毎フレームの画像処理ループ
function processVideo() {
  if (!isRunning || isMatched) return;

  try {
    cap.read(src);
    src.copyTo(dst);

    // グレースケール化とエッジ検出
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.Canny(gray, edges, 75, 200);

    // 輪郭（Contours）の抽出
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let maxArea = 0;
    let bestPoly = new cv.Mat();

    // 最も大きい四角形（枠）を探す
    for (let i = 0; i < contours.size(); ++i) {
      let cnt = contours.get(i);
      let area = cv.contourArea(cnt);
      if (area > 20000) { // 面積の閾値（カメラからの距離に合わせて調整）
        let approx = new cv.Mat();
        cv.approxPolyDP(cnt, approx, 0.02 * cv.arcLength(cnt, true), true);
        if (approx.rows === 4 && area > maxArea) {
          maxArea = area;
          approx.copyTo(bestPoly);
        }
        approx.delete();
      }
    }

    // 四角形が見つかった場合の処理
    if (bestPoly.rows === 4) {
      // 画面に検出した枠を描画（緑色）
      let pts = Array.from(bestPoly.data32S);
      for (let i = 0; i < 4; i++) {
        cv.line(dst, new cv.Point(pts[i*2], pts[i*2+1]), new cv.Point(pts[((i+1)%4)*2], pts[((i+1)%4)*2+1]), [0, 255, 0, 255], 3);
      }

      // コーナー座標を整列（左上、右上、右下、左下）
      let corners = sortCorners(pts);

      // 4. 射影変換（正面に補正）
      let warped = warpPerspective(src, corners, 400, 600); // 400x600の縦長サイズに補正

      // 5. テンプレートマッチング（特定エリアの判定）
      checkMatch(warped);

      warped.delete();
    }

    // メモリ解放と画面更新
    cv.imshow('canvasOutput', dst);
    bestPoly.delete(); contours.delete(); hierarchy.delete();
    
    requestAnimationFrame(processVideo);

  } catch (err) {
    console.error(err);
  }
}

// コーナーを[左上, 右上, 右下, 左下]の順に並び替える関数
function sortCorners(pts) {
  let points = [];
  for (let i = 0; i < 4; i++) points.push({x: pts[i*2], y: pts[i*2+1]});
  points.sort((a, b) => (a.y + a.x) - (b.y + b.x)); // x+yが最小=左上、最大=右下
  let tl = points[0];
  let br = points[3];
  let tr, bl;
  if (points[1].x > points[2].x) {
    tr = points[1]; bl = points[2];
  } else {
    tr = points[2]; bl = points[1];
  }
  return [tl, tr, br, bl];
}

// 射影変換を行う関数
function warpPerspective(srcMat, corners, width, height) {
  let srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    corners[0].x, corners[0].y, corners[1].x, corners[1].y,
    corners[2].x, corners[2].y, corners[3].x, corners[3].y
  ]);
  let dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0, width, 0, width, height, 0, height
  ]);
  
  let M = cv.getPerspectiveTransform(srcPts, dstPts);
  let warped = new cv.Mat();
  let dsize = new cv.Size(width, height);
  cv.warpPerspective(srcMat, warped, M, dsize);
  
  srcPts.delete(); dstPts.delete(); M.delete();
  return warped;
}

// 補正された画像から特定エリアを切り抜いて照合する関数
function checkMatch(warpedMat) {
  let warpedGray = new cv.Mat();
  cv.cvtColor(warpedMat, warpedGray, cv.COLOR_RGBA2GRAY);

  // 例: 紙の左上 (X:20, Y:20) から 幅100x高さ100 の範囲を抽出 (正解エリア)
  let roiRect = new cv.Rect(20, 20, 100, 100); 
  let roi = warpedGray.roi(roiRect);

  // テンプレートマッチング
  let result = new cv.Mat();
  // マッチング手法: 類似度を0〜1で返す手法
  cv.matchTemplate(roi, templateMat, result, cv.TM_CCOEFF_NORMED);
  let minMax = cv.minMaxLoc(result);

  // 類似度が閾値（0.8 = 80%）を超えたら一致とみなす
  if (minMax.maxVal > 0.8) {
    isMatched = true; // ループ停止
    document.getElementById('status').innerText = '一致しました！';
    document.getElementById('success-ui').style.display = 'block';
  }

  warpedGray.delete(); roi.delete(); result.delete();
}

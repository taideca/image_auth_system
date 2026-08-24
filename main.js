let video, cap, src, dst, gray, edges;
let targetData = [];
let matchedNames = new Set();
let totalTargets = 0;
let isRunning = false;
let isCooldown = false;
let currentCorners = null; // 現在のフレームで検出されている枠の四隅座標

// 1. OpenCV.js読み込み完了時の処理
function onOpenCvReady() {
  cv['onRuntimeInitialized'] = () => {
    document.getElementById('status').innerText = '正解データを読み込み中...';
    loadTargetsAndStart();
  };
}

// 2. JSONファイルからリストを取得し、画像を読み込む
async function loadTargetsAndStart() {
  try {
    const response = await fetch('targets.json');
    const targetNames = await response.json();
    totalTargets = targetNames.length;
    updateScoreUI();

    for (const name of targetNames) {
      const path = `img/${name}.png`; 
      try {
        const mat = await loadImageAsMat(path);
        targetData.push({ name: name, mat: mat });
      } catch (err) {
        console.error("画像の読み込みに失敗しました:", path);
      }
    }

    if (targetData.length === 0) {
      document.getElementById('status').innerText = 'エラー: 正解画像が読み込めませんでした';
      return;
    }

    document.getElementById('status').innerText = 'カメラを起動中...';
    startCamera();
  } catch (err) {
    document.getElementById('status').innerText = '設定ファイル(targets.json)の読み込みに失敗しました';
    console.error(err);
  }
}

// 画像を読み込み、線画（エッジ）に変換する関数
function loadImageAsMat(url) {
  return new Promise((resolve, reject) => {
    let img = new Image();
    img.onload = () => {
      let mat = cv.imread(img);
      cv.cvtColor(mat, mat, cv.COLOR_RGBA2GRAY);
      cv.Canny(mat, mat, 50, 150); 
      resolve(mat);
    };
    img.onerror = () => reject(new Error("Load error"));
    img.src = url;
  });
}

// 3. カメラの起動
function startCamera() {
  video = document.getElementById('videoInput');
  // スマホ(特にiPhone)でカメラを確実に動かすため、JSからも属性を強制付与
  video.setAttribute('autoplay', '');
  video.setAttribute('muted', '');
  video.setAttribute('playsinline', '');

  navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
    .then(function(stream) {
      video.srcObject = stream;
      
      // 再生が開始されたことを確実に検知してから処理に進む
      video.play().then(() => {
        let checkVideoSize = setInterval(() => {
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            clearInterval(checkVideoSize);
            initOpenCV();
          }
        }, 100);
      }).catch(err => {
        document.getElementById('status').innerText = '動画の再生がブロックされました: ' + err;
      });
    })
    .catch(function(err) {
      document.getElementById('status').innerText = 'カメラのアクセスに失敗しました';
    });
}

// 4. 画像処理変数の初期化
function initOpenCV() {
  try {
    let vw = video.videoWidth;
    let vh = video.videoHeight;
    
    let canvas = document.getElementById('canvasOutput');
    canvas.width = vw;
    canvas.height = vh;

    src = new cv.Mat(vh, vw, cv.CV_8UC4);
    dst = new cv.Mat(vh, vw, cv.CV_8UC4);
    gray = new cv.Mat();
    edges = new cv.Mat();
    cap = new cv.VideoCapture(video);

    document.getElementById('status').innerText = '枠を映し、ボタンを押してください';
    isRunning = true;
    
    document.getElementById('capture-btn').addEventListener('click', handleCapture);
    
    requestAnimationFrame(processVideo);
  } catch (err) {
    document.getElementById('status').innerText = '初期化エラー: ' + err;
  }
}

// 5. 毎フレームの画像処理ループ（★フリーズ防止の安全装置を追加）
function processVideo() {
  if (!isRunning) return;

  // 毎回生成する変数は try-catch の外に出して、エラー時も必ずメモリ解放できるようにする
  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  let bestPoly = new cv.Mat();

  try {
    cap.read(src);

    // 【重要】もし映像が空（一瞬の読み込み遅れなど）の場合は、処理をスキップしてフリーズを防ぐ
    if (!src.empty()) {
      src.copyTo(dst);

      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.Canny(gray, edges, 75, 200);

      cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      let maxArea = 0;
      let found = false;

      for (let i = 0; i < contours.size(); ++i) {
        let cnt = contours.get(i);
        let area = cv.contourArea(cnt);
        if (area > 20000) { // 面積の閾値
          let approx = new cv.Mat();
          cv.approxPolyDP(cnt, approx, 0.02 * cv.arcLength(cnt, true), true);
          if (approx.rows === 4 && area > maxArea) {
            maxArea = area;
            approx.copyTo(bestPoly);
            found = true;
          }
          approx.delete();
        }
      }

      let btn = document.getElementById('capture-btn');

      if (found) {
        let pts = Array.from(bestPoly.data32S);
        for (let i = 0; i < 4; i++) {
          cv.line(dst, new cv.Point(pts[i*2], pts[i*2+1]), new cv.Point(pts[((i+1)%4)*2], pts[((i+1)%4)*2+1]), [0, 255, 0, 255], 3);
        }
        currentCorners = sortCorners(pts);
        if (!isCooldown) btn.disabled = false;
      } else {
        currentCorners = null;
        btn.disabled = true;
      }

      // 画面に描画
      cv.imshow('canvasOutput', dst);
    }
  } catch (err) {
    // もしエラーが起きても、画面上部に内容を表示してプログラム自体は止めない
    console.error(err);
    document.getElementById('status').innerText = 'カメラ処理中エラー: ' + err.message;
  } finally {
    // 正常でもエラーでも、必ずメモリを解放して次のフレームを呼び出す（絶対フリーズさせない）
    bestPoly.delete(); 
    contours.delete(); 
    hierarchy.delete();
    requestAnimationFrame(processVideo);
  }
}

// ==========================================
// 6. 「判定する」ボタンが押された時の処理
// ==========================================
function handleCapture() {
  if (!currentCorners || isCooldown) return;
  
  // 処理中はボタンを無効化
  isCooldown = true;
  document.getElementById('capture-btn').disabled = true;

  // ボタンを押した瞬間の映像(src)と枠の座標(currentCorners)を使って射影変換
  let warped = warpPerspective(src, currentCorners, 400, 600);
  
  // 画像の照合処理を実行
  checkMatch(warped);
  
  warped.delete();
}

// コーナー並び替え関数
function sortCorners(pts) {
  let points = [];
  for (let i = 0; i < 4; i++) points.push({x: pts[i*2], y: pts[i*2+1]});
  points.sort((a, b) => (a.y + a.x) - (b.y + b.x));
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

// 射影変換関数
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

// 7. 画像の照合処理（ボタン押下時に1回だけ実行される）
function checkMatch(warpedMat) {
  let warpedGray = new cv.Mat();
  cv.cvtColor(warpedMat, warpedGray, cv.COLOR_RGBA2GRAY);
  let roiRect = new cv.Rect(0, 0, 140, 140); 
  let roi = warpedGray.roi(roiRect);

  let roiEdges = new cv.Mat();
  cv.Canny(roi, roiEdges, 50, 150);

  let matchFound = false;

  for (let i = 0; i < targetData.length; i++) {
    let target = targetData[i];

    // すでに正解したものはスキップ
    if (matchedNames.has(target.name)) continue;

    let result = new cv.Mat();
    cv.matchTemplate(roiEdges, target.mat, result, cv.TM_CCOEFF_NORMED);
    let minMax = cv.minMaxLoc(result);
    result.delete();

    if (minMax.maxVal > 0.5) {
      matchedNames.add(target.name);
      updateScoreUI();
      showSuccessPopup(`「${target.name}」を発見！`, true);
      matchFound = true;
      break; 
    }
  }

  // もしどの画像とも一致しなかった場合の処理
  if (!matchFound) {
    showSuccessPopup('一致する画像がありません', false);
  }

  warpedGray.delete(); roi.delete(); roiEdges.delete();
}

function updateScoreUI() {
  document.getElementById('score-board').innerText = `正解数: ${matchedNames.size} / ${totalTargets}`;
}

// 成功・失敗のポップアップ表示（色を分けてわかりやすく）
function showSuccessPopup(message, isSuccess) {
  let ui = document.getElementById('success-ui');
  ui.innerText = message;
  
  // 成功時は緑、失敗時は赤にする
  ui.style.background = isSuccess ? 'rgba(76, 175, 80, 0.9)' : 'rgba(244, 67, 54, 0.9)';
  ui.style.display = 'block';

  if (matchedNames.size === totalTargets && isSuccess) {
    ui.innerText = "すべての画像を見つけました！クリア！";
    return; 
  }

  // 2秒後にポップアップを消す
  setTimeout(() => {
    ui.style.display = 'none';
    isCooldown = false; 
  }, 2000);
}

let video, cap, src, dst, gray, edges;
let targetData = []; // 読み込んだ画像データと名前をセットで管理
let matchedNames = new Set(); // 既に正解した画像の名前を記録（重複防止）
let totalTargets = 0;
let isRunning = false;
let isCooldown = false; // 連続でマッチング反応するのを防ぐフラグ

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
    // targets.json を取得
    const response = await fetch('targets.json');
    const targetNames = await response.json();
    totalTargets = targetNames.length;
    updateScoreUI();

    // 取得した名前リストをもとに画像を読み込む
    for (const name of targetNames) {
      const path = `img/${name}.png`; // フォルダと拡張子を自動付与
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

// 画像パスからcv.Matを生成する関数
function loadImageAsMat(url) {
  return new Promise((resolve, reject) => {
    let img = new Image();
    img.onload = () => {
      let mat = cv.imread(img);
      cv.cvtColor(mat, mat, cv.COLOR_RGBA2GRAY);
      resolve(mat);
    };
    img.onerror = () => reject(new Error("Load error"));
    img.src = url;
  });
}

// 3. カメラの起動
function startCamera() {
  video = document.getElementById('videoInput');
  navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 640, height: 480 }, audio: false })
    .then(function(stream) {
      video.srcObject = stream;
      video.play();
      video.oncanplay = () => {
        initOpenCV();
      };
    })
    .catch(function(err) {
      document.getElementById('status').innerText = 'カメラのアクセスに失敗しました';
    });
}

// 4. 画像処理変数の初期化とループ開始
function initOpenCV() {
  src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  gray = new cv.Mat();
  edges = new cv.Mat();
  cap = new cv.VideoCapture(video);

  document.getElementById('status').innerText = '枠をカメラに映してください';
  isRunning = true;
  requestAnimationFrame(processVideo);
}

// 5. 毎フレームの画像処理ループ
function processVideo() {
  if (!isRunning) return;

  try {
    cap.read(src);
    src.copyTo(dst);

    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.Canny(gray, edges, 75, 200);

    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let maxArea = 0;
    let bestPoly = new cv.Mat();

    for (let i = 0; i < contours.size(); ++i) {
      let cnt = contours.get(i);
      let area = cv.contourArea(cnt);
      if (area > 20000) {
        let approx = new cv.Mat();
        cv.approxPolyDP(cnt, approx, 0.02 * cv.arcLength(cnt, true), true);
        if (approx.rows === 4 && area > maxArea) {
          maxArea = area;
          approx.copyTo(bestPoly);
        }
        approx.delete();
      }
    }

    if (bestPoly.rows === 4) {
      let pts = Array.from(bestPoly.data32S);
      for (let i = 0; i < 4; i++) {
        cv.line(dst, new cv.Point(pts[i*2], pts[i*2+1]), new cv.Point(pts[((i+1)%4)*2], pts[((i+1)%4)*2+1]), [0, 255, 0, 255], 3);
      }
      let corners = sortCorners(pts);
      let warped = warpPerspective(src, corners, 400, 600);
      
      // クールダウン中でなければ判定処理を実行
      if (!isCooldown) {
        checkMatch(warped);
      }
      warped.delete();
    }

    cv.imshow('canvasOutput', dst);
    bestPoly.delete(); contours.delete(); hierarchy.delete();
    
    requestAnimationFrame(processVideo);
  } catch (err) {
    console.error(err);
  }
}

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

// 6. 画像の照合処理
function checkMatch(warpedMat) {
  let warpedGray = new cv.Mat();
  cv.cvtColor(warpedMat, warpedGray, cv.COLOR_RGBA2GRAY);
  let roiRect = new cv.Rect(20, 20, 100, 100); 
  let roi = warpedGray.roi(roiRect);

  for (let i = 0; i < targetData.length; i++) {
    let target = targetData[i];

    // すでに正解済みの画像はスキップ（重複判定防止）
    if (matchedNames.has(target.name)) continue;

    let result = new cv.Mat();
    cv.matchTemplate(roi, target.mat, result, cv.TM_CCOEFF_NORMED);
    let minMax = cv.minMaxLoc(result);
    result.delete();

    // 一致した時の処理
    if (minMax.maxVal > 0.8) {
      matchedNames.add(target.name); // 正解リストに追加
      updateScoreUI();
      showSuccessPopup(`「${target.name}」を発見！`);
      break; 
    }
  }
  warpedGray.delete(); roi.delete();
}

// HTMLのスコア表示を更新する関数
function updateScoreUI() {
  document.getElementById('score-board').innerText = `正解数: ${matchedNames.size} / ${totalTargets}`;
}

// 画面中央にテキストを表示し、一時的に判定をストップさせる関数
function showSuccessPopup(message) {
  isCooldown = true; // 連続判定をストップ
  let ui = document.getElementById('success-ui');
  ui.innerText = message;
  ui.style.display = 'block';

  // 全ての画像をコンプリートした場合
  if (matchedNames.size === totalTargets) {
    ui.innerText = "すべての画像を見つけました！クリア！";
    return; // クールダウンを解除せずそのまま終了
  }

  // 3秒後にポップアップを消し、判定を再開する
  setTimeout(() => {
    ui.style.display = 'none';
    isCooldown = false; 
  }, 3000);
}

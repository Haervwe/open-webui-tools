/**
 * 3D Dice Roller - Self-contained Three.js + cannon-es dice physics
 * Built as IIFE, exposed as window.DiceRoller3D
 */
import * as THREE from 'three';
import * as CANNON from 'cannon-es';
import { createDieGeometry } from './geometries.js';

// ─── Per-face number textures ────────────────────────────────────────────────

function createFaceTexture(number, faceColor, textColor = '#fff', size = 256) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  // Face background
  ctx.fillStyle = faceColor;
  ctx.fillRect(0, 0, size, size);

  // Subtle radial shading for depth
  const g = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size*0.6);
  g.addColorStop(0, 'rgba(255,255,255,0.06)');
  g.addColorStop(1, 'rgba(0,0,0,0.12)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);

  // Number text
  const label = String(number);
  ctx.fillStyle = textColor;
  ctx.font = `bold ${size * (label.length > 1 ? 0.35 : 0.45)}px system-ui, -apple-system, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0,0,0,0.6)';
  ctx.shadowBlur = size * 0.03;
  ctx.fillText(label, size / 2, size / 2);

  // Underline 6 and 9 to distinguish them
  if (number === 6 || number === 9) {
    const tw = ctx.measureText(label).width;
    ctx.fillRect(size/2 - tw/2, size/2 + size*0.2, tw, size*0.025);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

function getLabelsForDie(sides) {
  if (sides === 100) return ['00','10','20','30','40','50','60','70','80','90'];
  const arr = [];
  for (let i = 1; i <= sides; i++) arr.push(i);
  return arr;
}

function hexToRGB(hex) {
  const c = hex.replace('#', '');
  return new THREE.Color(
    parseInt(c.substr(0, 2), 16) / 255,
    parseInt(c.substr(2, 2), 16) / 255,
    parseInt(c.substr(4, 2), 16) / 255,
  );
}

/**
 * Build array of materials with numbered textures, one per face.
 */
function buildFaceMaterials(sides, diceColor) {
  const labels = getLabelsForDie(sides);
  return labels.map((num) => {
    const tex = createFaceTexture(num, diceColor, '#ffffff');
    return new THREE.MeshPhysicalMaterial({
      map: tex,
      metalness: 0.05,
      roughness: 0.45,
      clearcoat: 0.25,
      clearcoatRoughness: 0.3,
      // NOTE: no flatShading — we compute per-face normals manually below
    });
  });
}

/**
 * Convert indexed geometry to non-indexed, compute per-face normals,
 * set UV for each triangle face, and assign material groups.
 */
function prepareDieGeometry(geometry, sides) {
  // Convert to non-indexed so each triangle has isolated vertices
  let geo = geometry.getIndex() ? geometry.toNonIndexed() : geometry.clone();

  const posAttr = geo.getAttribute('position');
  const vertCount = posAttr.count;
  const triCount = vertCount / 3;

  // Compute flat (per-face) normals, ensuring they point outward
  const normalArr = new Float32Array(vertCount * 3);
  const tmp = new THREE.Vector3();
  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const centroid = new THREE.Vector3();
  const faceCenter = new THREE.Vector3();
  // Compute geometry centroid (should be near origin for all our dice)
  centroid.set(0, 0, 0);
  for (let i = 0; i < triCount; i++) {
    vA.fromBufferAttribute(posAttr, i * 3);
    vB.fromBufferAttribute(posAttr, i * 3 + 1);
    vC.fromBufferAttribute(posAttr, i * 3 + 2);
    // Face normal via cross product
    tmp.crossVectors(vB.clone().sub(vA), vC.clone().sub(vA)).normalize();
    // Ensure normal points away from centroid (outward)
    faceCenter.set(
      (vA.x + vB.x + vC.x) / 3,
      (vA.y + vB.y + vC.y) / 3,
      (vA.z + vB.z + vC.z) / 3,
    );
    if (tmp.dot(faceCenter.clone().sub(centroid)) < 0) {
      tmp.negate(); // flip inward normals
    }
    for (let v = 0; v < 3; v++) {
      normalArr[(i * 3 + v) * 3 + 0] = tmp.x;
      normalArr[(i * 3 + v) * 3 + 1] = tmp.y;
      normalArr[(i * 3 + v) * 3 + 2] = tmp.z;
    }
  }
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normalArr, 3));

  // Set up UVs per triangle — centered square crop (works for any polygon shape)
  const uvArr = new Float32Array(vertCount * 2);
  for (let i = 0; i < triCount; i++) {
    uvArr[i * 6 + 0] = 0.5;  uvArr[i * 6 + 1] = 0.95;
    uvArr[i * 6 + 2] = 0.05; uvArr[i * 6 + 3] = 0.05;
    uvArr[i * 6 + 4] = 0.95; uvArr[i * 6 + 5] = 0.05;
  }
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvArr, 2));

  // Assign material groups
  geo.clearGroups();
  if (sides === 6) {
    // BoxGeometry non-indexed: 12 tris, 2 per face → groups of 6 vertices
    for (let i = 0; i < 6; i++) geo.addGroup(i * 6, 6, i);
  } else if (sides === 12) {
    // DodecahedronGeometry(r,0): 36 tris, 3 per face → groups of 9 vertices
    for (let i = 0; i < 12; i++) geo.addGroup(i * 9, 9, i);
  } else if (sides === 10 || sides === 100) {
    // D10 custom: 20 tris total, 4 per kite-diamond face (2 upper + 2 lower triangles per face)
    // But we only have 10 labels, so group 2 tris per label
    for (let i = 0; i < 10; i++) geo.addGroup(i * 6, 6, i);
  } else {
    // Tetra(4 tris), Octa(8 tris), Icosa(20 tris): 1 tri = 1 face
    for (let i = 0; i < sides; i++) geo.addGroup(i * 3, 3, i);
  }

  return geo;
}



// ─── Edge wireframe material ────────────────────────────────────────────────

function createEdgeMaterial() {
  return new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.18,
  });
}

// ─── Main roller class ──────────────────────────────────────────────────────

class DiceRoller {
  constructor(container, options = {}) {
    this.container = container;
    this.diceColor = options.diceColor || '#c0392b';
    this.animMs = options.animMs || 1500;
    this.diceBodies = [];
    this.diceMeshes = [];
    this.settled = false;
    this.onSettled = null;
    this._initScene();
    this._initPhysics();
    this._animate = this._animate.bind(this);
  }

  _initScene() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;

    this.scene = new THREE.Scene();
    this.scene.background = null;

    // Camera angle: slightly tilted top-down for a tabletop feel
    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
    this.camera.position.set(0, 8, 5);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.3;
    this.container.appendChild(this.renderer.domElement);

    // Lighting
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));

    const key = new THREE.DirectionalLight(0xfff8e7, 1.8);
    key.position.set(4, 10, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 30;
    key.shadow.camera.left = -8;
    key.shadow.camera.right = 8;
    key.shadow.camera.top = 8;
    key.shadow.camera.bottom = -8;
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x8888ff, 0.35);
    fill.position.set(-4, 6, -3);
    this.scene.add(fill);

    const rim = new THREE.PointLight(0xff8844, 0.5, 20);
    rim.position.set(0, 8, -6);
    this.scene.add(rim);

    // Felt surface
    const felt = new THREE.Mesh(
      new THREE.PlaneGeometry(16, 16),
      new THREE.MeshStandardMaterial({ color: 0x1a472a, roughness: 0.92, metalness: 0 })
    );
    felt.rotation.x = -Math.PI / 2;
    felt.position.y = -0.01;
    felt.receiveShadow = true;
    this.scene.add(felt);

    // Wooden border ring
    const border = new THREE.Mesh(
      new THREE.RingGeometry(7, 7.6, 64),
      new THREE.MeshStandardMaterial({ color: 0x3d2817, roughness: 0.7, metalness: 0.2 })
    );
    border.rotation.x = -Math.PI / 2;
    border.position.y = 0.01;
    this.scene.add(border);
  }

  _initPhysics() {
    this.world = new CANNON.World({ gravity: new CANNON.Vec3(0, -25, 0) });
    this.world.broadphase = new CANNON.NaiveBroadphase();
    this.world.solver.iterations = 20;

    // Contact material: dice on felt — high friction so spheres stop quickly
    const diceMat = new CANNON.Material('dice');
    const feltMat = new CANNON.Material('felt');
    this.world.addContactMaterial(new CANNON.ContactMaterial(diceMat, feltMat, {
      friction: 0.85,       // high friction — stops sliding/spinning fast
      restitution: 0.15,    // low bounce — dice land and stay
    }));
    this.world.addContactMaterial(new CANNON.ContactMaterial(diceMat, diceMat, {
      friction: 0.5,
      restitution: 0.2,
    }));
    this.diceMat = diceMat;

    // Ground body (the felt table)
    const ground = new CANNON.Body({ mass: 0, material: feltMat, shape: new CANNON.Plane() });
    ground.quaternion.setFromEuler(-Math.PI / 2, 0, 0);
    this.world.addBody(ground);

    // Invisible walls to contain the dice
    const wallMat = new CANNON.Material('wall');
    this.world.addContactMaterial(new CANNON.ContactMaterial(diceMat, wallMat, {
      friction: 0.2,
      restitution: 0.4,
    }));
    [
      { pos: [4.5, 3, 0], rot: [0, -Math.PI / 2, 0] },
      { pos: [-4.5, 3, 0], rot: [0, Math.PI / 2, 0] },
      { pos: [0, 3, 4.5], rot: [Math.PI / 2, 0, Math.PI] },
      { pos: [0, 3, -4.5], rot: [-Math.PI / 2, 0, 0] },
    ].forEach(({ pos, rot }) => {
      const b = new CANNON.Body({ mass: 0, material: wallMat, shape: new CANNON.Plane() });
      b.position.set(...pos);
      b.quaternion.setFromEuler(...rot);
      this.world.addBody(b);
    });
  }

  addDie(sides) {
    const radius = sides <= 6 ? 0.6 : sides <= 10 ? 0.65 : 0.7;
    const { geometry: rawGeometry, shape } = createDieGeometry(sides, radius);

    // Convert to non-indexed with per-face UVs and material groups
    const geometry = prepareDieGeometry(rawGeometry, sides);

    // Number textures per face
    const faceMats = buildFaceMaterials(sides, this.diceColor);

    const mesh = new THREE.Mesh(geometry, faceMats);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    // Subtle edge wireframe (use original indexed geometry for clean edges)
    const edges = new THREE.EdgesGeometry(rawGeometry, 15);
    mesh.add(new THREE.LineSegments(edges, createEdgeMaterial()));

    this.scene.add(mesh);

    // Physics body — sphere shape needs HIGH damping to settle (no flat face to rest on)
    const body = new CANNON.Body({
      mass: 1.0,
      material: this.diceMat,
      shape,
      linearDamping: 0.5,   // kills sliding quickly on the felt
      angularDamping: 0.7,  // kills spinning — critical for sphere shapes
    });

    // Spawn at a random position, already low (just above table)
    const throwAngle = Math.random() * Math.PI * 2;
    const throwDist = 1.0 + Math.random() * 1.5;
    body.position.set(
      Math.cos(throwAngle) * throwDist,
      0.8 + Math.random() * 0.4, // low spawn — just above table surface
      Math.sin(throwAngle) * throwDist
    );

    // Random initial orientation
    body.quaternion.setFromEuler(
      Math.random() * Math.PI * 2,
      Math.random() * Math.PI * 2,
      Math.random() * Math.PI * 2
    );

    // Throw toward center — moderate force so dice tumble & roll, not fly
    const throwForce = 3 + Math.random() * 2.5;
    body.velocity.set(
      -Math.cos(throwAngle) * throwForce,
      0.8 + Math.random() * 0.6, // small upward bounce
      -Math.sin(throwAngle) * throwForce
    );

    // Moderate tumble spin — enough to look natural, damping kills it fast
    body.angularVelocity.set(
      (Math.random() - 0.5) * 8,
      (Math.random() - 0.5) * 6,
      (Math.random() - 0.5) * 8
    );

    this.world.addBody(body);
    this.diceBodies.push(body);
    this.diceMeshes.push(mesh);
  }

  roll(diceList) {
    this.settled = false;
    this._settleCounter = 0;
    diceList.forEach((d) => this.addDie(d.sides));
    this._startTime = performance.now();
    this._rafId = requestAnimationFrame(this._animate);
  }

  _animate(now) {
    // Sub-step physics for stability
    const substeps = 2;
    for (let i = 0; i < substeps; i++) {
      this.world.step(1 / 120);
    }

    // Sync Three.js meshes to physics
    for (let i = 0; i < this.diceBodies.length; i++) {
      this.diceMeshes[i].position.copy(this.diceBodies[i].position);
      this.diceMeshes[i].quaternion.copy(this.diceBodies[i].quaternion);
    }
    this.renderer.render(this.scene, this.camera);

    const elapsed = now - this._startTime;
    if (elapsed > 600) {
      let stopped = true;
      for (const b of this.diceBodies) {
        if (b.velocity.length() > 0.05 || b.angularVelocity.length() > 0.08) {
          stopped = false;
          break;
        }
      }
      this._settleCounter = stopped ? this._settleCounter + 1 : 0;
      if ((this._settleCounter > 30 || elapsed > this.animMs + 2000) && !this.settled) {
        this.settled = true;
        if (this.onSettled) this.onSettled();
      }
    }

    if (!this.settled || elapsed < this.animMs + 2500) {
      this._rafId = requestAnimationFrame(this._animate);
    }
  }

  destroy() {
    if (this._rafId) cancelAnimationFrame(this._rafId);
    this.renderer.dispose();
    this.scene.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach((m) => {
          m.dispose();
          if (m.map) m.map.dispose();
        });
      }
    });
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}

// ─── Public API ──────────────────────────────────────────────────────────────

function rollDice3D(config) {
  return new Promise((resolve) => {
    const {
      dice = [{ sides: 20, sign: 1 }],
      modifier = 0,
      notation = '1d20',
      diceColor = '#c0392b',
      animMs = 1500,
    } = config;

    // Pre-roll values (the physics is cosmetic, values are predetermined)
    const results = dice.map((d) => ({
      value: Math.floor(Math.random() * d.sides) + 1,
      sides: d.sides,
      sign: d.sign,
    }));

    // Overlay
    const overlay = document.createElement('div');
    overlay.style.cssText = `position:fixed;inset:0;z-index:999999;
      background:rgba(0,0,0,0.75);backdrop-filter:blur(14px);
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      font-family:system-ui,-apple-system,'Segoe UI',sans-serif;
      animation:dr3dFadeIn 0.3s ease;`;

    const style = document.createElement('style');
    style.textContent = `
      @keyframes dr3dFadeIn{0%{opacity:0}100%{opacity:1}}
      @keyframes dr3dSlideUp{0%{opacity:0;transform:translateY(20px)}100%{opacity:1;transform:translateY(0)}}
      @keyframes dr3dPulse{0%,100%{opacity:0.7}50%{opacity:1}}
    `;
    document.head.appendChild(style);

    // Header
    const header = document.createElement('div');
    header.style.cssText = 'text-align:center;margin-bottom:12px;color:#fff;animation:dr3dSlideUp 0.4s ease;';
    header.innerHTML = `<div style="font-size:9px;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;color:${diceColor};opacity:0.8;margin-bottom:4px;">Dice Roller</div><div style="font-size:20px;font-weight:700;letter-spacing:-0.3px;">${notation}</div>`;
    overlay.appendChild(header);

    // Canvas container
    const canvasWrap = document.createElement('div');
    const sz = Math.min(500, window.innerWidth * 0.92);
    canvasWrap.style.cssText = `width:${sz}px;height:${sz * 0.6}px;border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);box-shadow:0 12px 40px rgba(0,0,0,0.4);margin-bottom:16px;animation:dr3dSlideUp 0.5s ease 0.1s both;`;
    overlay.appendChild(canvasWrap);

    const resultsArea = document.createElement('div');
    resultsArea.style.cssText = 'animation:dr3dSlideUp 0.4s ease;min-height:60px;text-align:center;color:#fff;';
    overlay.appendChild(resultsArea);

    const btnArea = document.createElement('div');
    btnArea.style.cssText = 'display:flex;justify-content:center;gap:12px;margin-top:12px;animation:dr3dSlideUp 0.4s ease 0.2s both;';
    overlay.appendChild(btnArea);

    function makeBtn(label) {
      const b = document.createElement('button');
      b.textContent = label;
      b.style.cssText = 'background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.18);color:#fff;padding:14px 48px;border-radius:12px;font-size:15px;font-weight:700;cursor:pointer;transition:all 0.2s;font-family:inherit;min-width:160px;box-shadow:0 4px 16px rgba(0,0,0,0.2);';
      b.onmouseenter = () => { b.style.background = 'rgba(255,255,255,0.2)'; };
      b.onmouseleave = () => { b.style.background = 'rgba(255,255,255,0.12)'; };
      return b;
    }

    const rollBtn = makeBtn('\uD83C\uDFB2 Roll');
    btnArea.appendChild(rollBtn);

    let roller = null;

    rollBtn.onclick = () => {
      rollBtn.style.display = 'none';
      roller = new DiceRoller(canvasWrap, { diceColor, animMs });
      roller.onSettled = showResults;
      roller.roll(results.map((r) => ({ sides: r.sides })));
    };

    function showResults() {
      let total = modifier;
      let html = '<div style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-bottom:12px;">';
      results.forEach((r) => {
        total += r.value * r.sign;
        const isCrit = r.value === r.sides;
        const isFumble = r.value === 1 && r.sides >= 4;
        const bc = isCrit ? '#ffd700' : (isFumble ? '#ff4444' : 'rgba(255,255,255,0.15)');
        const glow = isCrit ? 'box-shadow:0 0 12px rgba(255,215,0,0.6);' : (isFumble ? 'box-shadow:0 0 12px rgba(255,68,68,0.4);' : '');
        const prefix = r.sign < 0 ? '-' : '';
        html += `<div style="display:inline-flex;align-items:center;justify-content:center;width:48px;height:48px;border-radius:10px;background:rgba(0,0,0,0.4);border:2px solid ${bc};font-size:20px;font-weight:700;color:#fff;${glow}">${prefix}${r.value}</div>`;
      });
      if (modifier !== 0) {
        const ms = modifier > 0 ? '+' : '';
        html += `<div style="display:inline-flex;align-items:center;justify-content:center;padding:0 14px;height:48px;border-radius:10px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);font-size:18px;font-weight:600;color:#ccc;">${ms}${modifier}</div>`;
      }
      html += '</div>';

      const isNat20 = results.some((r) => r.value === 20 && r.sides === 20);
      const isNat1 = results.some((r) => r.value === 1 && r.sides === 20);
      let tGlow = '', tLabel = '';
      if (isNat20) {
        tGlow = 'text-shadow:0 0 20px rgba(255,215,0,0.8);color:#ffd700;';
        tLabel = '<div style="font-size:11px;font-weight:800;letter-spacing:2px;color:#ffd700;margin-top:4px;animation:dr3dPulse 0.8s ease infinite alternate;">CRITICAL HIT</div>';
      } else if (isNat1) {
        tGlow = 'text-shadow:0 0 20px rgba(255,68,68,0.8);color:#ff6b6b;';
        tLabel = '<div style="font-size:11px;font-weight:800;letter-spacing:2px;color:#ff6b6b;margin-top:4px;">CRITICAL FAIL</div>';
      }

      html += `<div style="padding:14px;border-radius:12px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);"><div style="font-size:8px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;color:#888;margin-bottom:4px;">Total</div><div style="font-size:44px;font-weight:300;letter-spacing:-2px;${tGlow}">${total}</div>${tLabel}</div>`;
      resultsArea.innerHTML = html;

      btnArea.innerHTML = '';
      const acceptBtn = makeBtn('\u2714 Accept Roll');
      acceptBtn.style.animation = 'dr3dSlideUp 0.3s ease';
      btnArea.appendChild(acceptBtn);
      acceptBtn.onclick = () => {
        if (roller) roller.destroy();
        if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
        if (style.parentNode) style.parentNode.removeChild(style);
        resolve(JSON.stringify({
          notation,
          results: results.map((r) => ({ value: r.value, sides: r.sides, sign: r.sign })),
          modifier,
          total,
        }));
      };
    }

    document.body.appendChild(overlay);
  });
}

window.DiceRoller3D = { rollDice3D };
// Signal readiness for injectors that listen before onload fires
window.dispatchEvent(new CustomEvent('DiceRoller3DReady'));
export { rollDice3D };

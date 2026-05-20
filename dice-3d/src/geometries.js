/**
 * Polyhedral dice geometries for Three.js + cannon-es
 * Three.js: uses built-in polyhedra for clean rendering
 * Cannon-es: uses sphere approximations for stable physics
 */
import * as THREE from 'three';
import * as CANNON from 'cannon-es';

/**
 * Create a Three.js geometry and matching cannon-es shape for a given die type.
 * @param {number} sides - Number of sides (4, 6, 8, 10, 12, 20, 100)
 * @param {number} radius - Die radius
 * @returns {{ geometry: THREE.BufferGeometry, shape: CANNON.Shape }}
 */
export function createDieGeometry(sides, radius = 1) {
  let geometry;
  let shape;

  switch (sides) {
    case 4: {
      geometry = new THREE.TetrahedronGeometry(radius, 0);
      shape = new CANNON.Sphere(radius * 0.75);
      break;
    }
    case 6: {
      const s = radius * 1.15;
      geometry = new THREE.BoxGeometry(s, s, s);
      shape = new CANNON.Box(new CANNON.Vec3(s * 0.5, s * 0.5, s * 0.5));
      break;
    }
    case 8: {
      geometry = new THREE.OctahedronGeometry(radius, 0);
      shape = new CANNON.Sphere(radius * 0.85);
      break;
    }
    case 10:
    case 100: {
      geometry = makeD10Geometry(radius);
      shape = new CANNON.Sphere(radius * 0.88);
      break;
    }
    case 12: {
      geometry = new THREE.DodecahedronGeometry(radius, 0);
      shape = new CANNON.Sphere(radius * 0.92);
      break;
    }
    case 20: {
      geometry = new THREE.IcosahedronGeometry(radius, 0);
      shape = new CANNON.Sphere(radius * 0.9);
      break;
    }
    default: {
      const s = radius * 1.15;
      geometry = new THREE.BoxGeometry(s, s, s);
      shape = new CANNON.Box(new CANNON.Vec3(s * 0.5, s * 0.5, s * 0.5));
    }
  }

  return { geometry, shape };
}

function makeD10Geometry(radius) {
  const geo = new THREE.BufferGeometry();
  const positions = [];
  const normals = [];

  const topRing = [];
  const botRing = [];
  for (let i = 0; i < 5; i++) {
    const a1 = (i * 2 * Math.PI) / 5 - Math.PI / 2;
    const a2 = ((i * 2 + 1) * Math.PI) / 5 - Math.PI / 2;
    topRing.push(new THREE.Vector3(Math.cos(a1) * radius, radius * 0.3, Math.sin(a1) * radius));
    botRing.push(new THREE.Vector3(Math.cos(a2) * radius, -radius * 0.3, Math.sin(a2) * radius));
  }
  const top = new THREE.Vector3(0, radius * 0.95, 0);
  const bot = new THREE.Vector3(0, -radius * 0.95, 0);

  function addTri(a, b, c) {
    const ab = b.clone().sub(a);
    const ac = c.clone().sub(a);
    const n = new THREE.Vector3().crossVectors(ab, ac).normalize();
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    normals.push(n.x, n.y, n.z, n.x, n.y, n.z, n.x, n.y, n.z);
  }

  for (let i = 0; i < 5; i++) {
    const i2 = (i + 1) % 5;
    addTri(top, topRing[i], botRing[i]);
    addTri(top, botRing[i], topRing[i2]);
    addTri(bot, botRing[i2 === 0 ? 4 : i2 - 1], topRing[i2]);
    addTri(bot, topRing[i2], botRing[i]);
  }

  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  return geo;
}

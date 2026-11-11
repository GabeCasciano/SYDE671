// Interactive point cloud viewer. Each .point-cloud element loads data-src:
// N float32 xyz followed by N uint8 rgb.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

document.querySelectorAll('.point-cloud').forEach(init);

async function init(el) {
  const buf = await (await fetch(el.dataset.src)).arrayBuffer();
  const n = buf.byteLength / 15;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf, 0, n * 3), 3));
  geom.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(buf, n * 12, n * 3), 3, true));
  geom.computeBoundingSphere();
  const { center, radius } = geom.boundingSphere;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);
  scene.add(new THREE.Points(geom, new THREE.PointsMaterial({
    size: 2, sizeAttenuation: false, vertexColors: true })));

  // Start at the first camera's view: x right, y down, z forward
  const camera = new THREE.PerspectiveCamera(45, 1, radius / 100, radius * 100);
  camera.up.set(0, -1, 0);
  camera.position.set(center.x, center.y, center.z - radius * 2.8);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.domElement.style.display = 'block';
  el.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center);
  controls.update();
  controls.saveState();

  const render = () => renderer.render(scene, camera);
  controls.addEventListener('change', render);
  renderer.domElement.addEventListener('dblclick', () => controls.reset());

  new ResizeObserver(() => {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    render();
  }).observe(el);
}

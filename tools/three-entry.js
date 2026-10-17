// Bundled into vendor/three.bundle.js as a classic script (window.THREE), so the
// wallpaper also works when opened straight from disk (file://), where ES modules can't load.
export * from 'three';
export { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
export { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

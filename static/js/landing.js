(() => {
  'use strict';
  const page = document.querySelector('.landing-page');
  const canvas = document.getElementById('insightScene');
  if (!page || !canvas || page.dataset.insightInitialized) return;
  page.dataset.insightInitialized = 'true';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)');
  const control = document.getElementById('insightMotion');
  const events = new AbortController();
  const on = (target, type, handler, options = {}) => target.addEventListener(type, handler, { ...options, signal: events.signal });
  let paused = false;
  let disposed = false;
  let sceneApi = null;
  const motionAllowed = () => !paused && !reduced.matches;
  page.querySelectorAll('.hero-preview, .hero-preview > .flex.flex-col > div, #contact > div > p, .ai-insight-flow, footer > div').forEach(el => el.setAttribute('data-reveal', ''));
  const revealElements = [...page.querySelectorAll('[data-reveal]')];
  const revealGroups = new Map();
  revealElements.forEach(el => {
    const group = el.parentElement;
    const index = revealGroups.get(group) || 0;
    revealGroups.set(group, index + 1);
    el.style.setProperty('--reveal-delay', `${Math.min(index * 95, 380)}ms`);
  });
  page.querySelectorAll('.hero-preview > .flex.flex-col > div').forEach((el, index) => {
    el.style.setProperty('--reveal-delay', `${450 + index * 350}ms`);
  });
  if (control) control.hidden = false;
  page.querySelectorAll('.hero-preview, .feature-card, .ai-insight-float').forEach(el => el.setAttribute('data-tilt', ''));
  const tiltElements = [...page.querySelectorAll('[data-tilt]')];
  let observer;

  function updateMotion() {
    page.classList.toggle('motion-paused', !motionAllowed());
    control.textContent = reduced.matches ? 'Reduced motion enabled' : paused ? 'Resume animation' : 'Pause animation';
    control.disabled = reduced.matches;
    control.setAttribute('aria-pressed', String(!motionAllowed()));
    if (!motionAllowed()) {
      observer?.disconnect();
      revealElements.forEach(el => el.classList.remove('reveal-pending'));
      tiltElements.forEach(el => { el.style.removeProperty('--tilt-x'); el.style.removeProperty('--tilt-y'); });
    }
    sceneApi?.refresh();
  }
  on(control, 'click', () => { paused = !paused; updateMotion(); });
  on(reduced, 'change', updateMotion);
  updateMotion();

  if ('IntersectionObserver' in window && motionAllowed()) {
    observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.remove('reveal-pending');
      entry.target.classList.add('reveal-visible');
      observer.unobserve(entry.target);
    }), { threshold: 0.08 });
    revealElements.forEach(el => { el.classList.add('reveal-pending'); observer.observe(el); });
  }

  tiltElements.forEach(el => {
    on(el, 'pointermove', event => {
      if (!motionAllowed() || !finePointer.matches) return;
      const rect = el.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
      const depth = el.matches('.hero-preview') ? 9 : 14;
      el.style.setProperty('--tilt-x', `${(0.5 - y) * depth}deg`);
      el.style.setProperty('--tilt-y', `${(x - 0.5) * depth}deg`);
    }, { passive: true });
    on(el, 'pointerleave', () => { el.style.setProperty('--tilt-x', '0deg'); el.style.setProperty('--tilt-y', '0deg'); });
  });

  page.querySelectorAll('a[href^="#"]').forEach(link => on(link, 'click', event => {
    const target = document.getElementById(link.getAttribute('href').slice(1));
    if (!target || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: motionAllowed() ? 'smooth' : 'instant', block: 'start' });
    history.replaceState(null, '', link.getAttribute('href'));
    target.setAttribute('tabindex', '-1');
    target.focus({ preventScroll: true });
  }));

  on(window, 'pagehide', event => {
    if (event.persisted) return;
    disposed = true;
    observer?.disconnect();
    sceneApi?.dispose();
    events.abort();
  });
  on(window, 'pageshow', () => sceneApi?.refresh());

  async function initializeScene() {
    // Dynamic import keeps navigation and readable content independent of the CDN.
    const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js');
    if (disposed) return;
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false, powerPreference: 'low-power' });
    renderer.setClearColor(0x000000, 0);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 60);
    camera.position.z = 17;
    const world = new THREE.Group();
    scene.add(world);
    const count = matchMedia('(max-width: 767px)').matches ? 96 : 192;
    const perSide = count / 2;
    const positions = new Float32Array(count * 3);
    const arrangements = Array.from({ length: 4 }, () => new Float32Array(count * 3));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
    const pointMaterial = new THREE.PointsMaterial({ size: 0.045, transparent: true, opacity: 0.85, depthWrite: false });
    const nodes = new THREE.Points(geometry, pointMaterial);
    nodes.frustumCulled = false;
    world.add(nodes);

    const connections = [];
    for (let side = 0; side < 2; side++) {
      for (let i = 0; i < perSide; i++) {
        connections.push([side * perSide + i, side * perSide + (i + 1) % perSide]);
        if (i % 3 === 0) connections.push([side * perSide + i, side * perSide + (i + 13) % perSide]);
      }
    }
    const linePositions = new Float32Array(connections.length * 6);
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3).setUsage(THREE.DynamicDrawUsage));
    const lineMaterial = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.15, depthWrite: false });
    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    lines.frustumCulled = false;
    world.add(lines);

    const signalCount = 16;
    const signalPositions = new Float32Array(signalCount * 3);
    const signalGeometry = new THREE.BufferGeometry();
    signalGeometry.setAttribute('position', new THREE.BufferAttribute(signalPositions, 3).setUsage(THREE.DynamicDrawUsage));
    const signalMaterial = new THREE.PointsMaterial({ size: 0.075, transparent: true, opacity: 0.9, depthWrite: false });
    const signals = new THREE.Points(signalGeometry, signalMaterial);
    signals.frustumCulled = false;
    world.add(signals);

    const documentMaterial = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.45, depthWrite: false });
    const documentGeometry = new THREE.BufferGeometry();
    const strokes = [];
    const segment = (x1, y1, x2, y2) => strokes.push(x1, y1, 0, x2, y2, 0);
    segment(-0.65, -0.9, 0.65, -0.9); segment(0.65, -0.9, 0.65, 0.9);
    segment(0.65, 0.9, -0.65, 0.9); segment(-0.65, 0.9, -0.65, -0.9);
    for (let row = 0; row < 5; row++) segment(-0.4, 0.55 - row * 0.23, row === 4 ? 0.1 : 0.4, 0.55 - row * 0.23);
    documentGeometry.setAttribute('position', new THREE.Float32BufferAttribute(strokes, 3));
    const documents = Array.from({ length: 6 }, (_, i) => {
      const mesh = new THREE.LineSegments(documentGeometry, documentMaterial);
      mesh.rotation.set(0.12, i < 3 ? 0.4 : -0.4, i % 2 ? -0.08 : 0.08);
      world.add(mesh);
      return mesh;
    });
    const scanGeometry = new THREE.BufferGeometry();
    scanGeometry.setAttribute('position', new THREE.Float32BufferAttribute([-0.6, 0, 0.025, 0.6, 0, 0.025], 3));
    const scanMaterial = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.85, depthWrite: false });
    const scans = documents.map(doc => {
      const beam = new THREE.Line(scanGeometry, scanMaterial);
      doc.add(beam);
      return beam;
    });
    const barGeometry = new THREE.BoxGeometry(0.28, 1, 0.28);
    const barMaterial = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, wireframe: true, depthWrite: false });
    const bars = new THREE.InstancedMesh(barGeometry, barMaterial, 12);
    bars.frustumCulled = false;
    world.add(bars);
    const dummy = new THREE.Object3D();
    let width = 0, height = 0, anchors = [], progress = 0, targetProgress = 0;
    let time = 0, last = 0, raf = 0, contextLost = false;
    const pointer = { x: 0, y: 0 };
    const sections = [...page.querySelectorAll('[data-scene]')];
    const clamp = (n, a = 0, b = 1) => Math.min(b, Math.max(a, n));
    const mix = (a, b, t) => a + (b - a) * t;

    function measureAnchors() {
      anchors = sections.map((el, i) => i === 0 ? 0 : Math.max(0, el.getBoundingClientRect().top + scrollY - innerHeight * 0.35));
      readScroll();
    }
    function readScroll() {
      let stage = 0;
      while (stage < anchors.length - 2 && scrollY >= anchors[stage + 1]) stage++;
      targetProgress = stage + clamp((scrollY - anchors[stage]) / Math.max(1, anchors[stage + 1] - anchors[stage]));
    }
    function resize() {
      width = innerWidth; height = innerHeight;
      renderer.setPixelRatio(Math.min(devicePixelRatio || 1, width < 768 ? 1 : 1.5));
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      const halfWidth = Math.tan(THREE.MathUtils.degToRad(21)) * 17 * camera.aspect;
      const spread = Math.min(1, width / 950);
      for (let i = 0; i < count; i++) {
        const local = i % perSide;
        const side = i < perSide ? -1 : 1;
        const center = side * halfWidth * 0.77;
        const a = local * 2.399963;
        const y = 1 - 2 * (local + 0.5) / perSide;
        const r = Math.sqrt(1 - y * y);
        const values = [
          [center + ((local % 4) - 1.5) * 0.23 * spread, (Math.floor(local / 4) % 8) * 0.25 - 0.8, -1.5 + Math.floor(local / 32) * 0.55],
          [center + Math.cos(a) * r * 2 * spread, y * 3.3, Math.sin(a) * r * 2],
          [center + ((local % 6) - 2.5) * 0.42 * spread, -1.8 + Math.floor(local / 6) / (perSide / 6) * (1.3 + (local % 6) * 0.47), Math.sin(a) * 0.15],
          [center + Math.cos(a) * r * 1.25 * spread, y * 1.7, Math.sin(a) * r * 1.25]
        ];
        values.forEach((value, stage) => arrangements[stage].set(value, i * 3));
      }
      documents.forEach((doc, i) => {
        doc.position.set((i < 3 ? -1 : 1) * halfWidth * 0.77 + (i % 3 - 1) * 0.23, (i % 3 - 1) * 2.2, -0.5 - (i % 3) * 0.5);
        doc.scale.setScalar(width < 768 ? 0.7 : 1);
      });
      for (let i = 0; i < 12; i++) {
        const j = i % 6;
        const h = 1.3 + j * 0.47;
        dummy.position.set((i < 6 ? -1 : 1) * halfWidth * 0.77 + (j - 2.5) * 0.42 * spread, -1.8 + h / 2, 0);
        dummy.scale.set(spread, h, 1);
        dummy.updateMatrix(); bars.setMatrixAt(i, dummy.matrix);
      }
      bars.instanceMatrix.needsUpdate = true;
      measureAnchors();
      refresh();
    }
    function theme() {
      const tokens = getComputedStyle(page);
      const processing = tokens.getPropertyValue('--insight-processing').trim();
      const primary = tokens.getPropertyValue('--insight-primary').trim();
      pointMaterial.color.set(processing); lineMaterial.color.set(processing);
      documentMaterial.color.set(processing); signalMaterial.color.set(primary); barMaterial.color.set(primary);
      scanMaterial.color.set(primary);
      refresh();
    }
    function draw() {
      const p = reduced.matches ? 0 : progress;
      const stage = Math.min(2, Math.floor(p));
      const t = p - stage;
      const eased = t * t * t * (t * (t * 6 - 15) + 10);
      for (let i = 0; i < positions.length; i++) positions[i] = mix(arrangements[stage][i], arrangements[stage + 1][i], eased);
      const networkStrength = reduced.matches ? 0 : Math.max(0, 1 - Math.abs(p - 1));
      for (let i = 0; i < count; i++) {
        positions[i * 3 + 2] += Math.sin(time * 0.65 + i * 0.24) * 0.22 * networkStrength;
      }
      const halfWidth = Math.tan(THREE.MathUtils.degToRad(21)) * 17 * camera.aspect;
      const spread = Math.min(1, width / 950);
      for (let i = 0; i < 12; i++) {
        const j = i % 6;
        const growth = clamp((p - 1.15 - j * 0.045) / 0.6);
        const height = Math.max(0.015, (1.3 + j * 0.47) * (1 - Math.pow(1 - growth, 3)));
        dummy.position.set((i < 6 ? -1 : 1) * halfWidth * 0.77 + (j - 2.5) * 0.42 * spread, -1.8 + height / 2, 0);
        dummy.scale.set(spread, height, 1);
        dummy.updateMatrix();
        bars.setMatrixAt(i, dummy.matrix);
      }
      bars.instanceMatrix.needsUpdate = true;
      geometry.attributes.position.needsUpdate = true;
      connections.forEach(([a, b], i) => {
        for (let axis = 0; axis < 3; axis++) {
          linePositions[i * 6 + axis] = positions[a * 3 + axis];
          linePositions[i * 6 + 3 + axis] = positions[b * 3 + axis];
        }
      });
      lineGeometry.attributes.position.needsUpdate = true;
      for (let i = 0; i < signalCount; i++) {
        const [a, b] = connections[(i * 17) % connections.length];
        const travel = (time * 0.18 + i / signalCount) % 1;
        for (let axis = 0; axis < 3; axis++) signalPositions[i * 3 + axis] = mix(positions[a * 3 + axis], positions[b * 3 + axis], travel);
      }
      signalGeometry.attributes.position.needsUpdate = true;
      const dark = document.documentElement.classList.contains('dark');
      lineMaterial.opacity = (dark ? 0.23 : 0.15) * (0.25 + 0.75 * Math.max(0, 1 - Math.abs(p - 1)));
      documentMaterial.opacity = (dark ? 0.65 : 0.48) * clamp(1 - p);
      scanMaterial.opacity = (dark ? 0.9 : 0.7) * clamp(1 - p);
      scans.forEach((beam, i) => {
        const cycle = reduced.matches ? 0.5 : (time * 0.22 + i * 0.17) % 1;
        beam.position.y = 0.78 - cycle * 1.56;
        beam.scale.x = Math.min(1, Math.sin(cycle * Math.PI) * 3);
      });
      barMaterial.opacity = (dark ? 0.65 : 0.45) * Math.max(0, 1 - Math.abs(p - 2));
      world.rotation.y = reduced.matches ? 0 : Math.sin(time * 0.16) * 0.09 + Math.sin(p * Math.PI) * 0.12;
      world.rotation.x = reduced.matches ? 0 : Math.cos(time * 0.12) * 0.045;
      documents.forEach((doc, i) => {
        doc.rotation.y = (i < 3 ? 0.4 : -0.4) + (reduced.matches ? 0 : Math.sin(time * 0.55 + i) * 0.18);
        doc.rotation.x = reduced.matches ? 0.12 : 0.12 + Math.cos(time * 0.4 + i) * 0.08;
      });
      renderer.render(scene, camera);
    }
    function frame(now) {
      raf = 0;
      if (disposed || contextLost || document.hidden) return;
      const delta = Math.min((now - (last || now)) / 1000, 0.05);
      last = now;
      if (motionAllowed()) {
        time += delta;
        const ease = 1 - Math.exp(-delta * 5);
        progress = mix(progress, targetProgress, ease);
        camera.position.x = mix(camera.position.x, pointer.x * 0.22, ease);
        camera.position.y = mix(camera.position.y, pointer.y * 0.15, ease);
      }
      draw();
      if (motionAllowed()) raf = requestAnimationFrame(frame);
    }
    function refresh() {
      if (raf) cancelAnimationFrame(raf);
      raf = 0; last = 0;
      if (disposed || contextLost || document.hidden) return;
      if (reduced.matches) { camera.position.x = 0; camera.position.y = 0; }
      draw();
      if (motionAllowed()) raf = requestAnimationFrame(frame);
    }
    on(window, 'scroll', readScroll, { passive: true });
    on(window, 'resize', resize, { passive: true });
    on(window, 'pointermove', event => {
      if (!motionAllowed() || !finePointer.matches) return;
      pointer.x = event.clientX / width - 0.5;
      pointer.y = 0.5 - event.clientY / height;
    }, { passive: true });
    on(document.documentElement, 'pointerleave', () => { pointer.x = 0; pointer.y = 0; });
    on(document, 'visibilitychange', refresh);
    on(canvas, 'webglcontextlost', event => { event.preventDefault(); contextLost = true; cancelAnimationFrame(raf); page.classList.remove('scene-ready'); control.hidden = false; });
    on(canvas, 'webglcontextrestored', () => { contextLost = false; page.classList.add('scene-ready'); control.hidden = false; resize(); });
    const themeObserver = new MutationObserver(theme);
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    const sizeObserver = 'ResizeObserver' in window ? new ResizeObserver(measureAnchors) : null;
    sizeObserver?.observe(page);
    sceneApi = {
      refresh,
      dispose() {
        cancelAnimationFrame(raf); themeObserver.disconnect(); sizeObserver?.disconnect();
        [geometry, lineGeometry, signalGeometry, documentGeometry, barGeometry, scanGeometry].forEach(item => item.dispose());
        [pointMaterial, lineMaterial, signalMaterial, documentMaterial, barMaterial, scanMaterial].forEach(item => item.dispose());
        renderer.dispose();
      }
    };
    resize(); theme();
    document.fonts?.ready.then(() => { if (!disposed) measureAnchors(); });
    page.classList.add('scene-ready'); control.hidden = false;
  }
  initializeScene().catch(() => {
    sceneApi?.dispose();
    page.classList.remove('scene-ready');
    control.hidden = false;
  });
})();

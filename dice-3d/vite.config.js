import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/dice-roller.js'),
      name: 'DiceRoller3D',
      formats: ['iife'],
      fileName: () => 'dice-roller.js',
    },
    rollupOptions: {
      output: {
        // Inline everything into a single file
        inlineDynamicImports: true,
      },
    },
    minify: 'esbuild',
    outDir: 'dist',
    emptyOutDir: true,
  },
});

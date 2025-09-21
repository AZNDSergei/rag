const cron = require('node-cron');
const { spawn } = require('child_process');
const path = require('path');

const projectDir = __dirname;
const py = process.platform === 'win32' ? 'python' : 'python3';

function runUpdate() {
  const args = [
    path.join(projectDir, 'update_index.py'),
    '--source', 'knowledge-base',
    '--out', 'index_out',
    '--min_words', '30',
    '--max_words', '120'
  ];

  const p = spawn(py, args, { cwd: projectDir, stdio: 'inherit' });
  p.on('close', code => console.log('[scheduler] update exit code', code));
  p.on('error', err => console.error('[scheduler] spawn error', err));
}

// каждые 2 минуты
cron.schedule('*/2 * * * *', runUpdate);
console.log('Node scheduler started (every 2 minutes)…');

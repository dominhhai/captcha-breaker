const PythonShell = require('python-shell')

const PY_SCRIPT = 'src/predict.py'

module.exports = image_path => (new Promise((resolve, reject) => {
  let opts = { }
  if (image_path) {
    opts.args = ['--fname', image_path]
  }
  PythonShell.run(PY_SCRIPT, opts, (err, results) => {
    if (err) return reject(err)
    resolve(results)
  })
}))

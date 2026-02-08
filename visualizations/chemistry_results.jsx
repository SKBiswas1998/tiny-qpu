import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine, ScatterChart, Scatter, ComposedChart, Area } from "recharts";

const h2Curve = [{"d": 0.3, "hf": -0.593828, "fci": -0.601804, "jw": -0.601804, "corr": -0.007976}, {"d": 0.35, "hf": -0.780455, "fci": -0.789269, "jw": -0.789269, "corr": -0.008814}, {"d": 0.4, "hf": -0.904361, "fci": -0.91415, "jw": -0.91415, "corr": -0.009788}, {"d": 0.45, "hf": -0.987513, "fci": -0.998416, "jw": -0.998416, "corr": -0.010902}, {"d": 0.5, "hf": -1.042996, "fci": -1.05516, "jw": -1.05516, "corr": -0.012164}, {"d": 0.55, "hf": -1.079051, "fci": -1.09263, "jw": -1.09263, "corr": -0.013579}, {"d": 0.6, "hf": -1.101128, "fci": -1.116286, "jw": -1.116286, "corr": -0.015158}, {"d": 0.65, "hf": -1.112997, "fci": -1.129905, "jw": -1.129905, "corr": -0.016908}, {"d": 0.7, "hf": -1.117349, "fci": -1.136189, "jw": -1.136189, "corr": -0.01884}, {"d": 0.75, "hf": -1.116151, "fci": -1.137117, "jw": -1.137117, "corr": -0.020966}, {"d": 0.8, "hf": -1.11085, "fci": -1.134148, "jw": -1.134148, "corr": -0.023297}, {"d": 0.85, "hf": -1.102511, "fci": -1.128362, "jw": -1.128362, "corr": -0.025851}, {"d": 0.9, "hf": -1.091914, "fci": -1.12056, "jw": -1.12056, "corr": -0.028646}, {"d": 0.95, "hf": -1.079637, "fci": -1.111339, "jw": -1.111339, "corr": -0.031702}, {"d": 1.0, "hf": -1.066109, "fci": -1.10115, "jw": -1.10115, "corr": -0.035042}, {"d": 1.05, "hf": -1.051657, "fci": -1.090342, "jw": -1.090342, "corr": -0.038685}, {"d": 1.1, "hf": -1.036539, "fci": -1.079193, "jw": -1.079193, "corr": -0.042654}, {"d": 1.15, "hf": -1.020964, "fci": -1.06793, "jw": -1.06793, "corr": -0.046966}, {"d": 1.2, "hf": -1.005107, "fci": -1.056741, "jw": -1.056741, "corr": -0.051634}, {"d": 1.25, "hf": -0.989114, "fci": -1.045783, "jw": -1.045783, "corr": -0.056669}, {"d": 1.3, "hf": -0.973111, "fci": -1.035186, "jw": -1.035186, "corr": -0.062076}, {"d": 1.35, "hf": -0.957203, "fci": -1.025054, "jw": -1.025054, "corr": -0.067851}, {"d": 1.4, "hf": -0.941481, "fci": -1.015468, "jw": -1.015468, "corr": -0.073988}, {"d": 1.45, "hf": -0.926017, "fci": -1.006487, "jw": -1.006487, "corr": -0.08047}, {"d": 1.5, "hf": -0.910874, "fci": -0.998149, "jw": -0.998149, "corr": -0.087276}, {"d": 1.55, "hf": -0.896099, "fci": -0.990476, "jw": -0.990476, "corr": -0.094377}, {"d": 1.6, "hf": -0.881732, "fci": -0.983473, "jw": -0.983473, "corr": -0.10174}, {"d": 1.65, "hf": -0.867804, "fci": -0.97713, "jw": -0.97713, "corr": -0.109325}, {"d": 1.7, "hf": -0.854338, "fci": -0.971427, "jw": -0.971427, "corr": -0.117089}, {"d": 1.75, "hf": -0.841349, "fci": -0.966335, "jw": -0.966335, "corr": -0.124986}, {"d": 1.8, "hf": -0.828848, "fci": -0.961817, "jw": -0.961817, "corr": -0.132969}, {"d": 1.85, "hf": -0.816842, "fci": -0.957833, "jw": -0.957833, "corr": -0.140991}, {"d": 1.9, "hf": -0.805333, "fci": -0.954339, "jw": -0.954339, "corr": -0.149006}, {"d": 1.95, "hf": -0.794318, "fci": -0.95129, "jw": -0.95129, "corr": -0.156972}, {"d": 2.0, "hf": -0.783793, "fci": -0.948641, "jw": -0.948641, "corr": -0.164848}, {"d": 2.05, "hf": -0.773749, "fci": -0.94635, "jw": -0.94635, "corr": -0.172601}, {"d": 2.1, "hf": -0.764178, "fci": -0.944375, "jw": -0.944375, "corr": -0.180197}, {"d": 2.15, "hf": -0.755066, "fci": -0.942678, "jw": -0.942678, "corr": -0.187612}, {"d": 2.2, "hf": -0.746401, "fci": -0.941224, "jw": -0.941224, "corr": -0.194823}, {"d": 2.25, "hf": -0.738169, "fci": -0.939982, "jw": -0.939982, "corr": -0.201813}, {"d": 2.3, "hf": -0.730353, "fci": -0.938922, "jw": -0.938922, "corr": -0.208569}, {"d": 2.35, "hf": -0.722939, "fci": -0.938021, "jw": -0.938021, "corr": -0.215082}, {"d": 2.4, "hf": -0.71591, "fci": -0.937255, "jw": -0.937255, "corr": -0.221345}, {"d": 2.45, "hf": -0.70925, "fci": -0.936605, "jw": -0.936605, "corr": -0.227355}, {"d": 2.5, "hf": -0.702944, "fci": -0.936055, "jw": -0.936055, "corr": -0.233111}, {"d": 2.55, "hf": -0.696975, "fci": -0.935589, "jw": -0.935589, "corr": -0.238615}, {"d": 2.6, "hf": -0.691328, "fci": -0.935196, "jw": -0.935196, "corr": -0.243868}, {"d": 2.65, "hf": -0.685988, "fci": -0.934864, "jw": -0.934864, "corr": -0.248876}, {"d": 2.7, "hf": -0.680941, "fci": -0.934584, "jw": -0.934584, "corr": -0.253644}, {"d": 2.75, "hf": -0.676172, "fci": -0.934349, "jw": -0.934349, "corr": -0.258177}, {"d": 2.8, "hf": -0.671669, "fci": -0.934151, "jw": -0.934151, "corr": -0.262482}, {"d": 2.85, "hf": -0.667417, "fci": -0.933985, "jw": -0.933985, "corr": -0.266568}, {"d": 2.9, "hf": -0.663405, "fci": -0.933846, "jw": -0.933846, "corr": -0.270441}, {"d": 2.95, "hf": -0.659619, "fci": -0.933729, "jw": -0.933729, "corr": -0.27411}, {"d": 3.0, "hf": -0.656048, "fci": -0.933632, "jw": -0.933632, "corr": -0.277584}];

const lihCurve = [{"d": 1.0, "hf": -7.767362, "fci": -7.78446}, {"d": 1.1, "hf": -7.808743, "fci": -7.825537}, {"d": 1.2, "hf": -7.835616, "fci": -7.852431}, {"d": 1.3, "hf": -7.851954, "fci": -7.86914}, {"d": 1.4, "hf": -7.860539, "fci": -7.878454}, {"d": 1.5, "hf": -7.863358, "fci": -7.882362}, {"d": 1.6, "hf": -7.861865, "fci": -7.882324}, {"d": 1.7, "hf": -7.857145, "fci": -7.879434}, {"d": 1.8, "hf": -7.850019, "fci": -7.874524}, {"d": 1.9, "hf": -7.841112, "fci": -7.868241}, {"d": 2.0, "hf": -7.830906, "fci": -7.861088}, {"d": 2.1, "hf": -7.81977, "fci": -7.853463}, {"d": 2.2, "hf": -7.807994, "fci": -7.845684}, {"d": 2.3, "hf": -7.795804, "fci": -7.838005}, {"d": 2.4, "hf": -7.783382, "fci": -7.830632}, {"d": 2.5, "hf": -7.770874, "fci": -7.823724}, {"d": 2.6, "hf": -7.758404, "fci": -7.8174}, {"d": 2.7, "hf": -7.74608, "fci": -7.811735}, {"d": 2.8, "hf": -7.733991, "fci": -7.806763}, {"d": 2.9, "hf": -7.722219, "fci": -7.802478}, {"d": 3.0, "hf": -7.71083, "fci": -7.798843}, {"d": 3.1, "hf": -7.699881, "fci": -7.795799}, {"d": 3.2, "hf": -7.689416, "fci": -7.793274}, {"d": 3.3, "hf": -7.679469, "fci": -7.791198}, {"d": 3.4, "hf": -7.67006, "fci": -7.789499}, {"d": 3.5, "hf": -7.661202, "fci": -7.788115}];

const vqeHistory = [-0.537908, -0.513598, -0.540657, -0.547886, -0.613923, -0.387477, -0.733275, -0.742596, -0.119219, -0.744559, -0.841185, -1.053386, -1.079443, -1.11543, -1.116371, -1.116733, -1.116753, -1.116758, -1.116759, -1.116759, -1.116759, -1.116759];

const h2Spectrum = [-1.137284, -0.538205, -0.538205, -0.530773, -0.530773, -0.530773, -0.445616, -0.445616, -0.168352, 0.240035, 0.240035, 0.355521, 0.355521, 0.483143, 0.715104, 0.923179];

const pauliTerms = {"IIZI": -0.22343, "IIIZ": -0.22343, "IIZZ": 0.17441, "ZIII": 0.17141, "IZII": 0.17141, "ZZII": 0.16869, "ZIIZ": 0.16593, "IZZI": 0.16593, "ZIZI": 0.12063, "IZIZ": 0.12063, "IIII": -0.09707, "YXXY": 0.04530, "YYXX": -0.04530, "XXYY": -0.04530, "XYYX": 0.04530};

const TABS = ["H₂ Bond Curve", "LiH Bond Curve", "VQE Convergence", "Energy Spectrum", "Pauli Decomposition", "Summary"];

const fmt = (v, d=6) => typeof v === 'number' ? v.toFixed(d) : v;

const CustomTooltip = ({ active, payload, label, unit }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1a1a2e', border: '1px solid #3a3a5c', borderRadius: 8, padding: '10px 14px', fontSize: 12 }}>
      <div style={{ color: '#8888aa', marginBottom: 4 }}>{label} {unit || ''}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginTop: 2 }}>
          {p.name}: {fmt(p.value)} Ha
        </div>
      ))}
    </div>
  );
};

const StatCard = ({ label, value, sub, accent = '#6ee7b7' }) => (
  <div style={{
    background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
    border: '1px solid #2a2a4a',
    borderRadius: 12,
    padding: '16px 20px',
    minWidth: 140,
    flex: 1,
  }}>
    <div style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', letterSpacing: 1.5, marginBottom: 6 }}>{label}</div>
    <div style={{ fontSize: 22, fontWeight: 700, color: accent, fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
    {sub && <div style={{ fontSize: 11, color: '#6b7280', marginTop: 4 }}>{sub}</div>}
  </div>
);

export default function ChemistryDashboard() {
  const [tab, setTab] = useState(0);

  const vqeData = vqeHistory.map((e, i) => ({ step: i, energy: e }));
  const corrData = h2Curve.map(d => ({ ...d, corr: Math.abs(d.corr) * 1000 }));

  const spectrumData = [];
  const seen = {};
  h2Spectrum.forEach((e, i) => {
    const key = e.toFixed(4);
    if (!seen[key]) { seen[key] = { energy: e, deg: 0, idx: Object.keys(seen).length }; }
    seen[key].deg++;
  });
  Object.values(seen).forEach(s => spectrumData.push(s));

  const pauliData = Object.entries(pauliTerms)
    .map(([k, v]) => ({ term: k, coeff: v, absCoeff: Math.abs(v) }))
    .sort((a, b) => b.absCoeff - a.absCoeff);

  const minH2 = h2Curve.reduce((a, b) => a.fci < b.fci ? a : b);
  const minLiH = lihCurve.reduce((a, b) => a.fci < b.fci ? a : b);

  return (
    <div style={{
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      background: 'linear-gradient(180deg, #0f0f23 0%, #0a0a1a 100%)',
      color: '#e2e8f0',
      minHeight: '100vh',
      padding: 0,
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;500;700;900&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0f0f23; }
        ::-webkit-scrollbar-thumb { background: #3a3a5c; border-radius: 3px; }
      `}</style>

      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%)',
        borderBottom: '1px solid #2a2a4a',
        padding: '28px 32px 20px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 8 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10,
            background: 'linear-gradient(135deg, #6ee7b7, #3b82f6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 20, fontWeight: 900,
          }}>⚛</div>
          <div>
            <h1 style={{ margin: 0, fontSize: 24, fontFamily: "'Outfit', sans-serif", fontWeight: 900, letterSpacing: -0.5 }}>
              tiny-qpu <span style={{ color: '#6ee7b7' }}>Chemistry</span> Results
            </h1>
            <div style={{ fontSize: 12, color: '#6b7280', marginTop: 2 }}>
              Jordan-Wigner Transform · PySCF Integration · STO-3G Basis
            </div>
          </div>
        </div>
      </div>

      {/* Tab Bar */}
      <div style={{
        display: 'flex', gap: 0, padding: '0 24px',
        background: '#12122a',
        borderBottom: '1px solid #2a2a4a',
        overflowX: 'auto',
      }}>
        {TABS.map((t, i) => (
          <button key={i} onClick={() => setTab(i)} style={{
            padding: '12px 18px',
            background: 'none',
            border: 'none',
            borderBottom: tab === i ? '2px solid #6ee7b7' : '2px solid transparent',
            color: tab === i ? '#6ee7b7' : '#6b7280',
            fontSize: 13,
            fontWeight: tab === i ? 600 : 400,
            cursor: 'pointer',
            whiteSpace: 'nowrap',
            transition: 'all 0.2s',
          }}>{t}</button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: '24px 28px', maxWidth: 920, margin: '0 auto' }}>

        {/* H2 BOND CURVE */}
        {tab === 0 && (
          <div>
            <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
              <StatCard label="Equilibrium" value={`${minH2.d} Å`} sub="Bond length" />
              <StatCard label="FCI Energy" value={fmt(minH2.fci, 4)} sub="Hartree" accent="#3b82f6" />
              <StatCard label="HF Energy" value={fmt(minH2.hf, 4)} sub="Hartree" accent="#f59e0b" />
              <StatCard label="JW = FCI" value="✓ Exact" sub="All 55 points" accent="#6ee7b7" />
            </div>

            <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: '20px 16px 12px' }}>
              <h3 style={{ margin: '0 0 4px 8px', fontSize: 15, fontWeight: 600 }}>H₂ Potential Energy Surface</h3>
              <div style={{ fontSize: 11, color: '#6b7280', margin: '0 0 16px 8px' }}>
                JW diagonalization overlaps FCI exactly (orange dots hidden behind blue)
              </div>
              <ResponsiveContainer width="100%" height={360}>
                <ComposedChart data={h2Curve} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
                  <XAxis dataKey="d" stroke="#6b7280" fontSize={11} label={{ value: 'Bond Length (Å)', position: 'bottom', offset: 4, fill: '#6b7280', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" fontSize={11} label={{ value: 'Energy (Ha)', angle: -90, position: 'insideLeft', offset: 10, fill: '#6b7280', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip unit="Å" />} />
                  <Legend verticalAlign="top" height={36} />
                  <Line type="monotone" dataKey="hf" name="Hartree-Fock" stroke="#f59e0b" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="fci" name="FCI (Exact)" stroke="#3b82f6" dot={false} strokeWidth={2.5} />
                  <Line type="monotone" dataKey="jw" name="JW Transform" stroke="#f97316" dot={false} strokeWidth={1.5} strokeDasharray="6 3" />
                  <ReferenceLine x={0.74} stroke="#6ee7b7" strokeDasharray="4 4" label={{ value: 'r₀', fill: '#6ee7b7', fontSize: 11 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: '20px 16px 12px', marginTop: 16 }}>
              <h3 style={{ margin: '0 0 12px 8px', fontSize: 15, fontWeight: 600 }}>Electron Correlation Energy</h3>
              <ResponsiveContainer width="100%" height={240}>
                <ComposedChart data={corrData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
                  <XAxis dataKey="d" stroke="#6b7280" fontSize={11} label={{ value: 'Bond Length (Å)', position: 'bottom', offset: 4, fill: '#6b7280', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" fontSize={11} label={{ value: '|E_corr| (mHa)', angle: -90, position: 'insideLeft', offset: 10, fill: '#6b7280', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip unit="Å" />} />
                  <Area type="monotone" dataKey="corr" name="Correlation" fill="#6ee7b720" stroke="#6ee7b7" strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: '#6b7280', marginLeft: 8, marginTop: 4 }}>
                Correlation energy grows dramatically at stretched geometries — exactly where HF fails and quantum methods shine.
              </div>
            </div>
          </div>
        )}

        {/* LiH BOND CURVE */}
        {tab === 1 && (
          <div>
            <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
              <StatCard label="Molecule" value="LiH" sub="Lithium Hydride" />
              <StatCard label="Equilibrium" value={`${minLiH.d} Å`} sub="Bond length" />
              <StatCard label="FCI Energy" value={fmt(minLiH.fci, 4)} sub="Hartree" accent="#3b82f6" />
              <StatCard label="Basis" value="STO-3G" sub="6 orbitals" accent="#a78bfa" />
            </div>

            <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: '20px 16px 12px' }}>
              <h3 style={{ margin: '0 0 4px 8px', fontSize: 15, fontWeight: 600 }}>LiH Potential Energy Surface</h3>
              <div style={{ fontSize: 11, color: '#6b7280', margin: '0 0 16px 8px' }}>
                Full-space FCI vs Hartree-Fock (STO-3G, 6 molecular orbitals)
              </div>
              <ResponsiveContainer width="100%" height={360}>
                <ComposedChart data={lihCurve} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
                  <XAxis dataKey="d" stroke="#6b7280" fontSize={11} label={{ value: 'Bond Length (Å)', position: 'bottom', offset: 4, fill: '#6b7280', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" fontSize={11} domain={['auto', 'auto']} label={{ value: 'Energy (Ha)', angle: -90, position: 'insideLeft', offset: 10, fill: '#6b7280', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip unit="Å" />} />
                  <Legend verticalAlign="top" height={36} />
                  <Line type="monotone" dataKey="hf" name="Hartree-Fock" stroke="#f59e0b" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="fci" name="FCI (Exact)" stroke="#3b82f6" dot={false} strokeWidth={2.5} />
                  <ReferenceLine x={minLiH.d} stroke="#6ee7b7" strokeDasharray="4 4" label={{ value: 'r₀', fill: '#6ee7b7', fontSize: 11 }} />
                </ComposedChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: '#6b7280', marginLeft: 8, marginTop: 4 }}>
                The HF/FCI gap widens at dissociation — multi-reference character requires correlated methods.
              </div>
            </div>
          </div>
        )}

        {/* VQE CONVERGENCE */}
        {tab === 2 && (
          <div>
            <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
              <StatCard label="Initial" value={fmt(vqeHistory[0], 4)} sub="Hartree" accent="#ef4444" />
              <StatCard label="Converged" value={fmt(vqeHistory[vqeHistory.length-1], 4)} sub="Hartree" accent="#6ee7b7" />
              <StatCard label="FCI Target" value="-1.1373" sub="Hartree" accent="#3b82f6" />
              <StatCard label="Iterations" value={vqeHistory.length} sub="L-BFGS-B" accent="#a78bfa" />
            </div>

            <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: '20px 16px 12px' }}>
              <h3 style={{ margin: '0 0 4px 8px', fontSize: 15, fontWeight: 600 }}>VQE Energy Convergence on H₂</h3>
              <div style={{ fontSize: 11, color: '#6b7280', margin: '0 0 16px 8px' }}>
                8-parameter hardware-efficient ansatz · Parameter-shift gradients · 4 qubits
              </div>
              <ResponsiveContainer width="100%" height={360}>
                <ComposedChart data={vqeData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
                  <XAxis dataKey="step" stroke="#6b7280" fontSize={11} label={{ value: 'Optimization Step', position: 'bottom', offset: 4, fill: '#6b7280', fontSize: 11 }} />
                  <YAxis stroke="#6b7280" fontSize={11} domain={[-1.2, 0]} label={{ value: 'Energy (Ha)', angle: -90, position: 'insideLeft', offset: 10, fill: '#6b7280', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={-1.137284} stroke="#3b82f6" strokeDasharray="6 3" label={{ value: 'FCI', fill: '#3b82f6', fontSize: 11, position: 'right' }} />
                  <ReferenceLine y={-1.116759} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: 'HF', fill: '#f59e0b', fontSize: 11, position: 'right' }} />
                  <Line type="monotone" dataKey="energy" name="VQE Energy" stroke="#6ee7b7" dot={{ r: 3, fill: '#6ee7b7' }} strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: '#6b7280', marginLeft: 8, marginTop: 8, lineHeight: 1.6 }}>
                VQE converges to HF energy (−1.1168 Ha) with this ansatz. Reaching FCI (−1.1373 Ha) requires a UCCSD-type ansatz
                that can capture the double excitation |0011⟩ → |1100⟩. The 20.5 mHa gap is the correlation energy.
              </div>
            </div>
          </div>
        )}

        {/* ENERGY SPECTRUM */}
        {tab === 3 && (
          <div>
            <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
              <StatCard label="Qubits" value="4" sub="Spin-orbitals" />
              <StatCard label="Hilbert Space" value="2⁴ = 16" sub="Basis states" accent="#a78bfa" />
              <StatCard label="Ground State" value={fmt(h2Spectrum[0], 4)} sub="Hartree" accent="#6ee7b7" />
              <StatCard label="Gap" value={fmt(h2Spectrum[3] - h2Spectrum[0], 4)} sub="Ha (to 1st excited)" accent="#f59e0b" />
            </div>

            <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: '20px 16px 12px' }}>
              <h3 style={{ margin: '0 0 4px 8px', fontSize: 15, fontWeight: 600 }}>H₂ Qubit Hamiltonian Eigenspectrum</h3>
              <div style={{ fontSize: 11, color: '#6b7280', margin: '0 0 16px 8px' }}>
                All 16 eigenvalues of the 4-qubit JW Hamiltonian at r = 0.74 Å
              </div>
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={spectrumData.map((s, i) => ({ ...s, label: `E${i}`, color: i === 0 ? '#6ee7b7' : s.deg > 1 ? '#a78bfa' : '#3b82f6' }))} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
                  <XAxis dataKey="label" stroke="#6b7280" fontSize={11} />
                  <YAxis stroke="#6b7280" fontSize={11} label={{ value: 'Energy (Ha)', angle: -90, position: 'insideLeft', offset: 10, fill: '#6b7280', fontSize: 11 }} />
                  <Tooltip formatter={(v) => [`${fmt(v)} Ha`]} contentStyle={{ background: '#1a1a2e', border: '1px solid #3a3a5c', borderRadius: 8, fontSize: 12 }} />
                  <Bar dataKey="energy" name="Energy" radius={[4, 4, 0, 0]}>
                    {spectrumData.map((s, i) => (
                      <Cell key={i} fill={i === 0 ? '#6ee7b7' : s.deg > 1 ? '#a78bfa' : '#3b82f6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ display: 'flex', gap: 20, marginLeft: 8, marginTop: 4 }}>
                <span style={{ fontSize: 11, color: '#6ee7b7' }}>● Ground state</span>
                <span style={{ fontSize: 11, color: '#a78bfa' }}>● Degenerate</span>
                <span style={{ fontSize: 11, color: '#3b82f6' }}>● Non-degenerate</span>
              </div>

              <div style={{ marginTop: 16, background: '#0f0f23', borderRadius: 8, padding: 12, fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }}>
                <div style={{ color: '#6b7280', marginBottom: 8 }}>Eigenvalue Table:</div>
                {spectrumData.map((s, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', color: i === 0 ? '#6ee7b7' : '#c9d1d9' }}>
                    <span>E{i}</span>
                    <span>{fmt(s.energy)} Ha</span>
                    <span style={{ color: '#6b7280' }}>{s.deg > 1 ? `${s.deg}× degenerate` : 'non-degenerate'}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* PAULI DECOMPOSITION */}
        {tab === 4 && (
          <div>
            <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
              <StatCard label="Pauli Terms" value="15" sub="Non-zero coefficients" />
              <StatCard label="Qubit Ops" value="I, X, Y, Z" sub="Tensor products" accent="#a78bfa" />
              <StatCard label="Largest" value={fmt(Math.max(...pauliData.map(p => p.absCoeff)), 4)} sub="|coefficient|" accent="#f59e0b" />
            </div>

            <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: '20px 16px 12px' }}>
              <h3 style={{ margin: '0 0 4px 8px', fontSize: 15, fontWeight: 600 }}>H₂ Qubit Hamiltonian Decomposition</h3>
              <div style={{ fontSize: 11, color: '#6b7280', margin: '0 0 16px 8px' }}>
                H = Σᵢ cᵢ Pᵢ where Pᵢ are Pauli strings (tensor products of I, X, Y, Z)
              </div>
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={pauliData} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 50 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e3a" />
                  <XAxis type="number" stroke="#6b7280" fontSize={11} label={{ value: 'Coefficient', position: 'bottom', offset: -2, fill: '#6b7280', fontSize: 11 }} />
                  <YAxis type="category" dataKey="term" stroke="#6b7280" fontSize={11} width={48} tick={{ fontFamily: "'JetBrains Mono', monospace" }} />
                  <Tooltip formatter={(v) => [fmt(v)]} contentStyle={{ background: '#1a1a2e', border: '1px solid #3a3a5c', borderRadius: 8, fontSize: 12 }} />
                  <ReferenceLine x={0} stroke="#4a4a6a" />
                  <Bar dataKey="coeff" name="Coefficient" radius={[0, 4, 4, 0]}>
                    {pauliData.map((p, i) => {
                      const hasXY = p.term.includes('X') || p.term.includes('Y');
                      return <Cell key={i} fill={hasXY ? '#a78bfa' : p.coeff > 0 ? '#3b82f6' : '#ef4444'} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ display: 'flex', gap: 20, marginLeft: 8, marginTop: 4 }}>
                <span style={{ fontSize: 11, color: '#3b82f6' }}>● ZZ terms (classical)</span>
                <span style={{ fontSize: 11, color: '#ef4444' }}>● Z terms (field)</span>
                <span style={{ fontSize: 11, color: '#a78bfa' }}>● XY terms (entangling)</span>
              </div>
              <div style={{ fontSize: 11, color: '#6b7280', marginLeft: 8, marginTop: 8, lineHeight: 1.6 }}>
                The XY/YX terms (purple) encode electron hopping — these are the quantum terms that make the problem hard classically.
                ZZ terms encode electron-electron repulsion. The identity term is the nuclear repulsion + mean-field offset.
              </div>
            </div>
          </div>
        )}

        {/* SUMMARY */}
        {tab === 5 && (
          <div>
            <div style={{
              background: 'linear-gradient(135deg, #1a2a1a 0%, #12122a 100%)',
              border: '1px solid #2a4a2a',
              borderRadius: 12,
              padding: 24,
              marginBottom: 20,
            }}>
              <h3 style={{ margin: '0 0 12px', fontSize: 18, fontWeight: 700, color: '#6ee7b7' }}>Feature #2: PySCF Chemistry Integration ✓</h3>
              <div style={{ fontSize: 13, lineHeight: 1.8, color: '#c9d1d9' }}>
                <div style={{ marginBottom: 12 }}>
                  The <span style={{ color: '#6ee7b7', fontFamily: "'JetBrains Mono', monospace" }}>tiny_qpu.chemistry</span> module
                  provides a complete pipeline from molecular geometry to qubit Hamiltonian, enabling VQE simulations of real molecules.
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              {[
                { title: 'Jordan-Wigner Transform', items: ['Fermion → qubit mapping via Pauli algebra', 'a†ₚ = ½(Xₚ - iYₚ) ⊗ Z chain', 'Full 4-operator products for 2-body terms', 'Verified: fermionic anticommutation {aₚ, a†q} = δₚq'] },
                { title: 'Molecular Integrals', items: ['PySCF Hartree-Fock → MO integrals', 'Spatial → spin-orbital conversion', 'Chemist notation (pq|rs) → a†ₚa†ᵣaₛaq', 'Active space reduction (frozen core)'] },
                { title: 'Validation Results', items: ['H₂: JW = FCI at all 55 bond lengths', 'LiH: JW matches CASCI(2,2) energy', 'All Hamiltonians verified Hermitian', 'VQE converges on molecular Hamiltonians'] },
                { title: 'Test Coverage', items: ['40 tests pass on Windows (no PySCF)', '43 tests pass on Linux (with PySCF)', 'Pre-computed integral fixtures embedded', 'Pauli algebra + anticommutation verified'] },
              ].map((section, i) => (
                <div key={i} style={{
                  background: '#12122a',
                  border: '1px solid #2a2a4a',
                  borderRadius: 12,
                  padding: '16px 20px',
                }}>
                  <h4 style={{ margin: '0 0 10px', fontSize: 14, color: '#a78bfa' }}>{section.title}</h4>
                  {section.items.map((item, j) => (
                    <div key={j} style={{ fontSize: 12, color: '#9ca3af', padding: '3px 0', display: 'flex', gap: 8 }}>
                      <span style={{ color: '#6ee7b7' }}>›</span> {item}
                    </div>
                  ))}
                </div>
              ))}
            </div>

            <div style={{
              marginTop: 20,
              background: '#12122a',
              border: '1px solid #2a2a4a',
              borderRadius: 12,
              padding: '16px 20px',
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 12,
              color: '#9ca3af',
              lineHeight: 1.8,
            }}>
              <div style={{ color: '#6b7280', marginBottom: 8, fontFamily: "'Segoe UI', sans-serif", fontWeight: 600 }}>Key Numbers</div>
              <div><span style={{ color: '#6ee7b7' }}>H₂ FCI Energy:</span> −1.137284 Ha at r = 0.74 Å (STO-3G)</div>
              <div><span style={{ color: '#6ee7b7' }}>H₂ Correlation:</span> −20.5 mHa at equilibrium, −277.6 mHa at 3.0 Å</div>
              <div><span style={{ color: '#6ee7b7' }}>LiH FCI Energy:</span> −7.882362 Ha at r = 1.5 Å (STO-3G)</div>
              <div><span style={{ color: '#6ee7b7' }}>Qubit Hamiltonian:</span> 15 Pauli terms, 4 qubits (H₂)</div>
              <div><span style={{ color: '#6ee7b7' }}>JW Accuracy:</span> Exact match to FCI (error {'<'} 10⁻⁶ Ha)</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

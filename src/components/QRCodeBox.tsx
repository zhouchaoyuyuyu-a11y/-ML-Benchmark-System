// Deterministic QR-style identity mark rendered as SVG. Encodes the id into a
// stable visual pattern; when a real QR service or WeChat QR image is
// configured (qr_code_url / CDN), pages display that image instead.

function hashBits(seed: string, count: number): boolean[] {
  const bits: boolean[] = [];
  let h = 2166136261;
  for (let i = 0; i < count; i++) {
    const c = seed.charCodeAt(i % seed.length);
    h ^= c + i;
    h = Math.imul(h, 16777619);
    bits.push(((h >>> ((i % 4) * 8)) & 0xff) > 127);
  }
  return bits;
}

export default function QRCodeBox({ seed, label, size = 160 }: { seed: string; label?: string; size?: number }) {
  const n = 13;
  const cell = Math.floor(size / n);
  const bits = hashBits(seed, n * n);

  const finder = (x: number, y: number) => (
    <g key={`f${x}${y}`}>
      <rect x={x * cell} y={y * cell} width={cell * 3} height={cell * 3} fill="none" stroke="#ece9e2" strokeWidth={cell * 0.6} />
      <rect x={(x + 1) * cell + cell * 0.2} y={(y + 1) * cell + cell * 0.2} width={cell * 0.6} height={cell * 0.6} fill="#c8a962" />
    </g>
  );

  return (
    <figure className="inline-flex flex-col items-center gap-2">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${n * cell} ${n * cell}`}
        role="img"
        aria-label={label ?? `Identity code ${seed}`}
        className="rounded-lg border border-hairline bg-ink p-1"
      >
        {bits.map((on, i) => {
          const x = i % n;
          const y = Math.floor(i / n);
          const inFinder = (x < 4 && y < 4) || (x > n - 5 && y < 4) || (x < 4 && y > n - 5);
          if (!on || inFinder) return null;
          return <rect key={i} x={x * cell + 1} y={y * cell + 1} width={cell - 2} height={cell - 2} fill="#ece9e2" opacity={0.9} />;
        })}
        {finder(0.5, 0.5)}
        {finder(n - 3.5, 0.5)}
        {finder(0.5, n - 3.5)}
      </svg>
      {label && <figcaption className="text-xs text-mist">{label}</figcaption>}
    </figure>
  );
}

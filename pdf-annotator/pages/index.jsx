import { useState, useRef, useEffect } from 'react';
import Head from 'next/head';

export default function PDFAnnotator() {
  const [fileName, setFileName] = useState('');
  const [pdfBytes, setPdfBytes] = useState(null);
  const [pages, setPages] = useState([]);
  const [annotations, setAnnotations] = useState([]);
  const [pending, setPending] = useState(null); // {pageIndex, cssX, cssY, pdfX, pdfY}
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);
  const pdfjsRef = useRef(null);
  const annotIdRef = useRef(0);

  useEffect(() => {
    import('pdfjs-dist').then((pdfjs) => {
      pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';
      pdfjsRef.current = pdfjs;
    });
  }, []);

  const loadPDF = async (file) => {
    if (!file || !pdfjsRef.current) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please upload a PDF file');
      return;
    }
    setLoading(true);
    setAnnotations([]);
    setPending(null);
    setFileName(file.name);

    const buffer = await file.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    setPdfBytes(bytes);

    const pdf = await pdfjsRef.current.getDocument({ data: bytes.slice().buffer }).promise;
    const SCALE = 1.5;
    const rendered = [];

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const vp = page.getViewport({ scale: SCALE });
      const canvas = document.createElement('canvas');
      canvas.width = vp.width;
      canvas.height = vp.height;
      await page.render({ canvasContext: canvas.getContext('2d'), viewport: vp }).promise;
      rendered.push({
        dataUrl: canvas.toDataURL('image/jpeg', 0.92),
        canvasW: vp.width,
        canvasH: vp.height,
        pdfW: vp.width / SCALE,
        pdfH: vp.height / SCALE,
        scale: SCALE,
      });
    }

    setPages(rendered);
    setLoading(false);
  };

  const handleFileInput = (e) => loadPDF(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    loadPDF(e.dataTransfer.files[0]);
  };

  // Click on a page: compute both CSS coords (for badge display) and PDF coords (for annotation writing)
  const handlePageClick = (e, pageIndex) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const cssY = e.clientY - rect.top;

    const pg = pages[pageIndex];
    // CSS size may differ from canvas size if browser scales the image
    const cssScaleX = rect.width / pg.canvasW;
    const cssScaleY = rect.height / pg.canvasH;
    const canvasX = cssX / cssScaleX;
    const canvasY = cssY / cssScaleY;

    // PDF uses bottom-left origin; canvas uses top-left
    const pdfX = canvasX / pg.scale;
    const pdfY = pg.pdfH - canvasY / pg.scale;

    setPending({ pageIndex, cssX, cssY, pdfX, pdfY });
    setInputText('');
    setTimeout(() => inputRef.current?.focus(), 20);
  };

  const confirmAnnotation = () => {
    if (!inputText.trim() || !pending) return;
    setAnnotations((prev) => [
      ...prev,
      { id: annotIdRef.current++, ...pending, text: inputText.trim() },
    ]);
    setPending(null);
    setInputText('');
  };

  const cancelPending = () => {
    setPending(null);
    setInputText('');
  };

  const removeAnnotation = (id) => {
    setAnnotations((prev) => prev.filter((a) => a.id !== id));
  };

  const download = async () => {
    if (!pdfBytes || annotations.length === 0) return;

    const { PDFDocument, PDFName, PDFString, PDFNumber, PDFArray } = await import('pdf-lib');
    const doc = await PDFDocument.load(pdfBytes);
    const docPages = doc.getPages();

    for (const ann of annotations) {
      const page = docPages[ann.pageIndex];
      const textW = Math.max(40, ann.text.length * 11 + 16);
      const textH = 24;

      // Build the FreeText annotation dict
      const annotObj = doc.context.obj({
        Type: PDFName.of('Annot'),
        Subtype: PDFName.of('FreeText'),
        Rect: doc.context.obj([
          ann.pdfX,
          ann.pdfY - textH / 2,
          ann.pdfX + textW,
          ann.pdfY + textH / 2,
        ]),
        Contents: PDFString.of(ann.text),
        // Default appearance: Helvetica 14pt, red color
        DA: PDFString.of('/Helvetica 14 Tf 1 0 0 rg'),
        F: PDFNumber.of(4), // Print flag
        Border: doc.context.obj([0, 0, 0]),
      });

      const annotRef = doc.context.register(annotObj);
      const Annots = PDFName.of('Annots');

      // Append to the page's Annots array (or create it)
      let annotsArray = null;
      try {
        annotsArray = page.node.lookup(Annots, PDFArray);
      } catch (_) {
        annotsArray = null;
      }

      if (annotsArray) {
        annotsArray.push(annotRef);
      } else {
        page.node.set(Annots, doc.context.obj([annotRef]));
      }
    }

    const saved = await doc.save();
    const blob = new Blob([saved], { type: 'application/pdf' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName.replace(/\.pdf$/i, '_annotated.pdf');
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <Head>
        <title>PDF Annotator</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <style jsx global>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: #f0f2f5;
          color: #1a1a1a;
        }
      `}</style>

      <div style={{ minHeight: '100vh', padding: '28px 20px' }}>
        <div style={{ maxWidth: '1000px', margin: '0 auto' }}>

          {/* Header */}
          <div style={{ marginBottom: '28px' }}>
            <h1 style={{ fontSize: '26px', fontWeight: 700, letterSpacing: '-0.5px' }}>
              PDF Annotator
            </h1>
            <p style={{ color: '#666', marginTop: '6px', fontSize: '15px' }}>
              Upload a PDF, click anywhere on a page to place a provider code, then download.
            </p>
          </div>

          {/* Upload area */}
          {!pages.length && !loading && (
            <div
              onDrop={handleDrop}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onClick={() => document.getElementById('file-input').click()}
              style={{
                border: `2px dashed ${dragOver ? '#2563eb' : '#ccc'}`,
                borderRadius: '14px',
                padding: '72px 40px',
                textAlign: 'center',
                background: dragOver ? '#eff6ff' : '#fff',
                cursor: 'pointer',
                transition: 'all 0.15s',
                boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
              }}
            >
              <div style={{ fontSize: '52px', marginBottom: '14px' }}>📄</div>
              <div style={{ fontSize: '18px', fontWeight: 600 }}>
                Drop a PDF here or click to upload
              </div>
              <div style={{ color: '#999', marginTop: '8px', fontSize: '14px' }}>
                Annotations are saved as FreeText and readable by PyMuPDF
              </div>
              <input
                id="file-input"
                type="file"
                accept=".pdf"
                onChange={handleFileInput}
                style={{ display: 'none' }}
              />
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div style={{ textAlign: 'center', padding: '72px', color: '#666' }}>
              <div style={{ fontSize: '36px', marginBottom: '14px' }}>⏳</div>
              <div style={{ fontSize: '16px' }}>Rendering PDF pages…</div>
            </div>
          )}

          {/* Toolbar */}
          {pages.length > 0 && (
            <div style={{
              display: 'flex', gap: '12px', alignItems: 'center',
              marginBottom: '16px', padding: '12px 16px',
              background: '#fff', borderRadius: '10px',
              boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
            }}>
              <span style={{ fontWeight: 600, flex: 1, fontSize: '15px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                📄 {fileName}
              </span>
              <span style={{ color: '#666', fontSize: '13px', whiteSpace: 'nowrap' }}>
                {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
              </span>
              <button
                onClick={() => { setPages([]); setPdfBytes(null); setAnnotations([]); setFileName(''); setPending(null); }}
                style={{
                  padding: '6px 14px', borderRadius: '6px', border: '1px solid #ddd',
                  cursor: 'pointer', background: '#fff', fontSize: '13px',
                }}
              >
                New PDF
              </button>
              <button
                onClick={download}
                disabled={annotations.length === 0}
                style={{
                  padding: '7px 18px', borderRadius: '6px', border: 'none',
                  cursor: annotations.length > 0 ? 'pointer' : 'not-allowed',
                  background: annotations.length > 0 ? '#2563eb' : '#b0b8c8',
                  color: '#fff', fontWeight: 600, fontSize: '13px',
                  transition: 'background 0.15s',
                }}
              >
                Download PDF
              </button>
            </div>
          )}

          {/* Hint banner */}
          {pages.length > 0 && (
            <div style={{
              background: '#fefce8', border: '1px solid #fde68a',
              borderRadius: '8px', padding: '10px 14px',
              fontSize: '13px', marginBottom: '20px', color: '#854d0e',
            }}>
              💡 Click anywhere on a page to add an annotation — e.g.{' '}
              <strong>7</strong>, <strong>1/7</strong>, or <strong>1/7/SRNA</strong>.
              Press Enter to confirm, Escape to cancel.
            </div>
          )}

          {/* PDF Pages */}
          {pages.map((pg, pageIndex) => (
            <div key={pageIndex} style={{ marginBottom: '28px' }}>
              <div style={{ fontSize: '12px', color: '#999', marginBottom: '6px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Page {pageIndex + 1}
              </div>
              <div
                style={{
                  position: 'relative',
                  display: 'inline-block',
                  cursor: 'crosshair',
                  boxShadow: '0 2px 16px rgba(0,0,0,0.12)',
                  borderRadius: '2px',
                  lineHeight: 0,
                  maxWidth: '100%',
                  overflow: 'hidden',
                }}
                onClick={(e) => handlePageClick(e, pageIndex)}
              >
                <img
                  src={pg.dataUrl}
                  width={pg.canvasW}
                  height={pg.canvasH}
                  style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
                  draggable={false}
                  alt={`Page ${pageIndex + 1}`}
                />

                {/* Placed annotations */}
                {annotations
                  .filter((a) => a.pageIndex === pageIndex)
                  .map((a) => (
                    <div
                      key={a.id}
                      style={{
                        position: 'absolute',
                        left: a.cssX,
                        top: a.cssY,
                        transform: 'translate(-50%, -50%)',
                        background: '#dc2626',
                        color: '#fff',
                        fontSize: '13px',
                        fontWeight: 700,
                        padding: '3px 8px',
                        borderRadius: '4px',
                        whiteSpace: 'nowrap',
                        boxShadow: '0 1px 6px rgba(0,0,0,0.25)',
                        userSelect: 'none',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '5px',
                        pointerEvents: 'all',
                      }}
                    >
                      {a.text}
                      <span
                        onClick={(e) => { e.stopPropagation(); removeAnnotation(a.id); }}
                        style={{
                          cursor: 'pointer',
                          opacity: 0.75,
                          fontSize: '15px',
                          lineHeight: 1,
                          fontWeight: 400,
                        }}
                        title="Remove"
                      >
                        ×
                      </span>
                    </div>
                  ))}

                {/* Pending annotation input */}
                {pending && pending.pageIndex === pageIndex && (
                  <div
                    style={{
                      position: 'absolute',
                      left: Math.min(pending.cssX, pg.canvasW - 240),
                      top: pending.cssY + 14,
                      background: '#fff',
                      border: '2px solid #2563eb',
                      borderRadius: '8px',
                      padding: '8px 10px',
                      zIndex: 30,
                      boxShadow: '0 4px 20px rgba(0,0,0,0.18)',
                      display: 'flex',
                      gap: '6px',
                      alignItems: 'center',
                    }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <input
                      ref={inputRef}
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') confirmAnnotation();
                        if (e.key === 'Escape') cancelPending();
                      }}
                      placeholder="e.g. 7 or 1/7"
                      style={{
                        width: '120px',
                        padding: '5px 8px',
                        border: '1px solid #d1d5db',
                        borderRadius: '5px',
                        fontSize: '14px',
                        outline: 'none',
                        fontFamily: 'monospace',
                      }}
                    />
                    <button
                      onClick={confirmAnnotation}
                      style={{
                        padding: '5px 12px', borderRadius: '5px',
                        background: '#2563eb', color: '#fff',
                        border: 'none', cursor: 'pointer', fontWeight: 700,
                        fontSize: '14px',
                      }}
                    >
                      ✓
                    </button>
                    <button
                      onClick={cancelPending}
                      style={{
                        padding: '5px 9px', borderRadius: '5px',
                        background: '#f3f4f6', color: '#555',
                        border: '1px solid #ddd', cursor: 'pointer',
                        fontSize: '14px',
                      }}
                    >
                      ✕
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

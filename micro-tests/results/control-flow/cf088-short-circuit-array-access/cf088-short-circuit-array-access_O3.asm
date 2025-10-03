
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf088-short-circuit-array-access/cf088-short-circuit-array-access_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z24test_array_short_circuitPii>:
100000360: aa0003e8    	mov	x8, x0
100000364: 52800000    	mov	w0, #0x0                ; =0
100000368: b4000148    	cbz	x8, 0x100000390 <__Z24test_array_short_circuitPii+0x30>
10000036c: 7100043f    	cmp	w1, #0x1
100000370: 5400010b    	b.lt	0x100000390 <__Z24test_array_short_circuitPii+0x30>
100000374: 52800000    	mov	w0, #0x0                ; =0
100000378: 2a0103e9    	mov	w9, w1
10000037c: b840450a    	ldr	w10, [x8], #0x4
100000380: 3400008a    	cbz	w10, 0x100000390 <__Z24test_array_short_circuitPii+0x30>
100000384: 0b000140    	add	w0, w10, w0
100000388: f1000529    	subs	x9, x9, #0x1
10000038c: 54ffff81    	b.ne	0x10000037c <__Z24test_array_short_circuitPii+0x1c>
100000390: d65f03c0    	ret

0000000100000394 <_main>:
100000394: 52800060    	mov	w0, #0x3                ; =3
100000398: d65f03c0    	ret

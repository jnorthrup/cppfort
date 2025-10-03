
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf088-short-circuit-array-access/cf088-short-circuit-array-access_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z24test_array_short_circuitPii>:
100000360: aa0003e8    	mov	x8, x0
100000364: 52800000    	mov	w0, #0x0                ; =0
100000368: b4000188    	cbz	x8, 0x100000398 <__Z24test_array_short_circuitPii+0x38>
10000036c: 7100043f    	cmp	w1, #0x1
100000370: 5400014b    	b.lt	0x100000398 <__Z24test_array_short_circuitPii+0x38>
100000374: d2800009    	mov	x9, #0x0                ; =0
100000378: 52800000    	mov	w0, #0x0                ; =0
10000037c: 2a0103ea    	mov	w10, w1
100000380: b869790b    	ldr	w11, [x8, x9, lsl #2]
100000384: 340000ab    	cbz	w11, 0x100000398 <__Z24test_array_short_circuitPii+0x38>
100000388: 0b000160    	add	w0, w11, w0
10000038c: 91000529    	add	x9, x9, #0x1
100000390: eb0a013f    	cmp	x9, x10
100000394: 54ffff63    	b.lo	0x100000380 <__Z24test_array_short_circuitPii+0x20>
100000398: d65f03c0    	ret

000000010000039c <_main>:
10000039c: 52800060    	mov	w0, #0x3                ; =3
1000003a0: d65f03c0    	ret

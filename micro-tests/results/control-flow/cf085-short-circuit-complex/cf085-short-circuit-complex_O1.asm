
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf085-short-circuit-complex/cf085-short-circuit-complex_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z6check1Ri>:
100000360: b9400008    	ldr	w8, [x0]
100000364: 11000508    	add	w8, w8, #0x1
100000368: b9000008    	str	w8, [x0]
10000036c: 52800020    	mov	w0, #0x1                ; =1
100000370: d65f03c0    	ret

0000000100000374 <__Z6check2Ri>:
100000374: b9400008    	ldr	w8, [x0]
100000378: 11002908    	add	w8, w8, #0xa
10000037c: b9000008    	str	w8, [x0]
100000380: 52800000    	mov	w0, #0x0                ; =0
100000384: d65f03c0    	ret

0000000100000388 <__Z6check3Ri>:
100000388: b9400008    	ldr	w8, [x0]
10000038c: 11019108    	add	w8, w8, #0x64
100000390: b9000008    	str	w8, [x0]
100000394: 52800020    	mov	w0, #0x1                ; =1
100000398: d65f03c0    	ret

000000010000039c <__Z26test_complex_short_circuitv>:
10000039c: 52800160    	mov	w0, #0xb                ; =11
1000003a0: d65f03c0    	ret

00000001000003a4 <_main>:
1000003a4: 52800160    	mov	w0, #0xb                ; =11
1000003a8: d65f03c0    	ret

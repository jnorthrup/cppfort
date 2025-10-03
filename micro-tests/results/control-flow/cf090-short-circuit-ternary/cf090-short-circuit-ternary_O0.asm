
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf090-short-circuit-ternary/cf090-short-circuit-ternary_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z26test_short_circuit_ternaryi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: 71000108    	subs	w8, w8, #0x0
100000374: 5400010d    	b.le	0x100000394 <__Z26test_short_circuit_ternaryi+0x34>
100000378: 14000001    	b	0x10000037c <__Z26test_short_circuit_ternaryi+0x1c>
10000037c: b9400be8    	ldr	w8, [sp, #0x8]
100000380: 11000508    	add	w8, w8, #0x1
100000384: b9000be8    	str	w8, [sp, #0x8]
100000388: 52800148    	mov	w8, #0xa                ; =10
10000038c: b90003e8    	str	w8, [sp]
100000390: 14000007    	b	0x1000003ac <__Z26test_short_circuit_ternaryi+0x4c>
100000394: b9400be8    	ldr	w8, [sp, #0x8]
100000398: 11001508    	add	w8, w8, #0x5
10000039c: b9000be8    	str	w8, [sp, #0x8]
1000003a0: 52800288    	mov	w8, #0x14               ; =20
1000003a4: b90003e8    	str	w8, [sp]
1000003a8: 14000001    	b	0x1000003ac <__Z26test_short_circuit_ternaryi+0x4c>
1000003ac: b94003e8    	ldr	w8, [sp]
1000003b0: b90007e8    	str	w8, [sp, #0x4]
1000003b4: b9400be0    	ldr	w0, [sp, #0x8]
1000003b8: 910043ff    	add	sp, sp, #0x10
1000003bc: d65f03c0    	ret

00000001000003c0 <_main>:
1000003c0: d10083ff    	sub	sp, sp, #0x20
1000003c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c8: 910043fd    	add	x29, sp, #0x10
1000003cc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d0: 528000a0    	mov	w0, #0x5                ; =5
1000003d4: 97ffffe3    	bl	0x100000360 <__Z26test_short_circuit_ternaryi>
1000003d8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003dc: 910083ff    	add	sp, sp, #0x20
1000003e0: d65f03c0    	ret

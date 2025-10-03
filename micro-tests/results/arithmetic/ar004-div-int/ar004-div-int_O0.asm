
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar004-div-int/ar004-div-int_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_divii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b90007e1    	str	w1, [sp, #0x4]
10000036c: b94007e8    	ldr	w8, [sp, #0x4]
100000370: 35000088    	cbnz	w8, 0x100000380 <__Z8test_divii+0x20>
100000374: 14000001    	b	0x100000378 <__Z8test_divii+0x18>
100000378: b9000fff    	str	wzr, [sp, #0xc]
10000037c: 14000006    	b	0x100000394 <__Z8test_divii+0x34>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: b94007e9    	ldr	w9, [sp, #0x4]
100000388: 1ac90d08    	sdiv	w8, w8, w9
10000038c: b9000fe8    	str	w8, [sp, #0xc]
100000390: 14000001    	b	0x100000394 <__Z8test_divii+0x34>
100000394: b9400fe0    	ldr	w0, [sp, #0xc]
100000398: 910043ff    	add	sp, sp, #0x10
10000039c: d65f03c0    	ret

00000001000003a0 <_main>:
1000003a0: d10083ff    	sub	sp, sp, #0x20
1000003a4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a8: 910043fd    	add	x29, sp, #0x10
1000003ac: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003b0: 52800280    	mov	w0, #0x14               ; =20
1000003b4: 52800081    	mov	w1, #0x4                ; =4
1000003b8: 97ffffea    	bl	0x100000360 <__Z8test_divii>
1000003bc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003c0: 910083ff    	add	sp, sp, #0x20
1000003c4: d65f03c0    	ret

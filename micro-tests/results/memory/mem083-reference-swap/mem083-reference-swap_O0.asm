
/Users/jim/work/cppfort/micro-tests/results/memory/mem083-reference-swap/mem083-reference-swap_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z4swapRiS_>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: f9000fe0    	str	x0, [sp, #0x18]
100000368: f9000be1    	str	x1, [sp, #0x10]
10000036c: f9400fe8    	ldr	x8, [sp, #0x18]
100000370: b9400108    	ldr	w8, [x8]
100000374: b9000fe8    	str	w8, [sp, #0xc]
100000378: f9400be8    	ldr	x8, [sp, #0x10]
10000037c: b9400108    	ldr	w8, [x8]
100000380: f9400fe9    	ldr	x9, [sp, #0x18]
100000384: b9000128    	str	w8, [x9]
100000388: b9400fe8    	ldr	w8, [sp, #0xc]
10000038c: f9400be9    	ldr	x9, [sp, #0x10]
100000390: b9000128    	str	w8, [x9]
100000394: 910083ff    	add	sp, sp, #0x20
100000398: d65f03c0    	ret

000000010000039c <_main>:
10000039c: d10083ff    	sub	sp, sp, #0x20
1000003a0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a4: 910043fd    	add	x29, sp, #0x10
1000003a8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003ac: 910023e0    	add	x0, sp, #0x8
1000003b0: 528000a8    	mov	w8, #0x5                ; =5
1000003b4: b9000be8    	str	w8, [sp, #0x8]
1000003b8: 910013e1    	add	x1, sp, #0x4
1000003bc: 52800148    	mov	w8, #0xa                ; =10
1000003c0: b90007e8    	str	w8, [sp, #0x4]
1000003c4: 97ffffe7    	bl	0x100000360 <__Z4swapRiS_>
1000003c8: b9400be0    	ldr	w0, [sp, #0x8]
1000003cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d0: 910083ff    	add	sp, sp, #0x20
1000003d4: d65f03c0    	ret

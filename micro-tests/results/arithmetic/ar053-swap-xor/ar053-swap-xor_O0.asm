
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar053-swap-xor/ar053-swap-xor_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z13test_swap_xorRiS_>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: f90007e0    	str	x0, [sp, #0x8]
100000368: f90003e1    	str	x1, [sp]
10000036c: f94003e8    	ldr	x8, [sp]
100000370: b940010a    	ldr	w10, [x8]
100000374: f94007e9    	ldr	x9, [sp, #0x8]
100000378: b9400128    	ldr	w8, [x9]
10000037c: 4a0a0108    	eor	w8, w8, w10
100000380: b9000128    	str	w8, [x9]
100000384: f94007e8    	ldr	x8, [sp, #0x8]
100000388: b940010a    	ldr	w10, [x8]
10000038c: f94003e9    	ldr	x9, [sp]
100000390: b9400128    	ldr	w8, [x9]
100000394: 4a0a0108    	eor	w8, w8, w10
100000398: b9000128    	str	w8, [x9]
10000039c: f94003e8    	ldr	x8, [sp]
1000003a0: b940010a    	ldr	w10, [x8]
1000003a4: f94007e9    	ldr	x9, [sp, #0x8]
1000003a8: b9400128    	ldr	w8, [x9]
1000003ac: 4a0a0108    	eor	w8, w8, w10
1000003b0: b9000128    	str	w8, [x9]
1000003b4: 910043ff    	add	sp, sp, #0x10
1000003b8: d65f03c0    	ret

00000001000003bc <_main>:
1000003bc: d10083ff    	sub	sp, sp, #0x20
1000003c0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c4: 910043fd    	add	x29, sp, #0x10
1000003c8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003cc: 910023e0    	add	x0, sp, #0x8
1000003d0: 528000a8    	mov	w8, #0x5                ; =5
1000003d4: b9000be8    	str	w8, [sp, #0x8]
1000003d8: 910013e1    	add	x1, sp, #0x4
1000003dc: 52800148    	mov	w8, #0xa                ; =10
1000003e0: b90007e8    	str	w8, [sp, #0x4]
1000003e4: 97ffffdf    	bl	0x100000360 <__Z13test_swap_xorRiS_>
1000003e8: b9400be0    	ldr	w0, [sp, #0x8]
1000003ec: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f0: 910083ff    	add	sp, sp, #0x20
1000003f4: d65f03c0    	ret

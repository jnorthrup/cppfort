
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar068-clamp-function/ar068-clamp-function_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z10test_clampiii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b90007e1    	str	w1, [sp, #0x4]
10000036c: b90003e2    	str	w2, [sp]
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: b94007e9    	ldr	w9, [sp, #0x4]
100000378: 6b090108    	subs	w8, w8, w9
10000037c: 540000aa    	b.ge	0x100000390 <__Z10test_clampiii+0x30>
100000380: 14000001    	b	0x100000384 <__Z10test_clampiii+0x24>
100000384: b94007e8    	ldr	w8, [sp, #0x4]
100000388: b9000fe8    	str	w8, [sp, #0xc]
10000038c: 1400000c    	b	0x1000003bc <__Z10test_clampiii+0x5c>
100000390: b9400be8    	ldr	w8, [sp, #0x8]
100000394: b94003e9    	ldr	w9, [sp]
100000398: 6b090108    	subs	w8, w8, w9
10000039c: 540000ad    	b.le	0x1000003b0 <__Z10test_clampiii+0x50>
1000003a0: 14000001    	b	0x1000003a4 <__Z10test_clampiii+0x44>
1000003a4: b94003e8    	ldr	w8, [sp]
1000003a8: b9000fe8    	str	w8, [sp, #0xc]
1000003ac: 14000004    	b	0x1000003bc <__Z10test_clampiii+0x5c>
1000003b0: b9400be8    	ldr	w8, [sp, #0x8]
1000003b4: b9000fe8    	str	w8, [sp, #0xc]
1000003b8: 14000001    	b	0x1000003bc <__Z10test_clampiii+0x5c>
1000003bc: b9400fe0    	ldr	w0, [sp, #0xc]
1000003c0: 910043ff    	add	sp, sp, #0x10
1000003c4: d65f03c0    	ret

00000001000003c8 <_main>:
1000003c8: d10083ff    	sub	sp, sp, #0x20
1000003cc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003d0: 910043fd    	add	x29, sp, #0x10
1000003d4: 52800001    	mov	w1, #0x0                ; =0
1000003d8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003dc: 528001e0    	mov	w0, #0xf                ; =15
1000003e0: 52800142    	mov	w2, #0xa                ; =10
1000003e4: 97ffffdf    	bl	0x100000360 <__Z10test_clampiii>
1000003e8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003ec: 910083ff    	add	sp, sp, #0x20
1000003f0: d65f03c0    	ret

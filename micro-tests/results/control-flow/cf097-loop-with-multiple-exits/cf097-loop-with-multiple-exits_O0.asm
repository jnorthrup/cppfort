
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf097-loop-with-multiple-exits/cf097-loop-with-multiple-exits_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_multiple_exitsi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b90007ff    	str	wzr, [sp, #0x4]
10000036c: 14000001    	b	0x100000370 <__Z19test_multiple_exitsi+0x10>
100000370: b94007e8    	ldr	w8, [sp, #0x4]
100000374: 71019108    	subs	w8, w8, #0x64
100000378: 540004aa    	b.ge	0x10000040c <__Z19test_multiple_exitsi+0xac>
10000037c: 14000001    	b	0x100000380 <__Z19test_multiple_exitsi+0x20>
100000380: b94007e8    	ldr	w8, [sp, #0x4]
100000384: b9400be9    	ldr	w9, [sp, #0x8]
100000388: 6b090108    	subs	w8, w8, w9
10000038c: 540000a1    	b.ne	0x1000003a0 <__Z19test_multiple_exitsi+0x40>
100000390: 14000001    	b	0x100000394 <__Z19test_multiple_exitsi+0x34>
100000394: b94007e8    	ldr	w8, [sp, #0x4]
100000398: b9000fe8    	str	w8, [sp, #0xc]
10000039c: 1400001e    	b	0x100000414 <__Z19test_multiple_exitsi+0xb4>
1000003a0: b94007e8    	ldr	w8, [sp, #0x4]
1000003a4: b94007e9    	ldr	w9, [sp, #0x4]
1000003a8: 1b097d08    	mul	w8, w8, w9
1000003ac: b9400be9    	ldr	w9, [sp, #0x8]
1000003b0: 6b090108    	subs	w8, w8, w9
1000003b4: 540000ad    	b.le	0x1000003c8 <__Z19test_multiple_exitsi+0x68>
1000003b8: 14000001    	b	0x1000003bc <__Z19test_multiple_exitsi+0x5c>
1000003bc: 12800008    	mov	w8, #-0x1               ; =-1
1000003c0: b9000fe8    	str	w8, [sp, #0xc]
1000003c4: 14000014    	b	0x100000414 <__Z19test_multiple_exitsi+0xb4>
1000003c8: b94007e8    	ldr	w8, [sp, #0x4]
1000003cc: 5280014a    	mov	w10, #0xa               ; =10
1000003d0: 1aca0d09    	sdiv	w9, w8, w10
1000003d4: 1b0a7d29    	mul	w9, w9, w10
1000003d8: 6b090108    	subs	w8, w8, w9
1000003dc: 350000e8    	cbnz	w8, 0x1000003f8 <__Z19test_multiple_exitsi+0x98>
1000003e0: 14000001    	b	0x1000003e4 <__Z19test_multiple_exitsi+0x84>
1000003e4: b94007e8    	ldr	w8, [sp, #0x4]
1000003e8: 7100c908    	subs	w8, w8, #0x32
1000003ec: 5400006d    	b.le	0x1000003f8 <__Z19test_multiple_exitsi+0x98>
1000003f0: 14000001    	b	0x1000003f4 <__Z19test_multiple_exitsi+0x94>
1000003f4: 14000006    	b	0x10000040c <__Z19test_multiple_exitsi+0xac>
1000003f8: 14000001    	b	0x1000003fc <__Z19test_multiple_exitsi+0x9c>
1000003fc: b94007e8    	ldr	w8, [sp, #0x4]
100000400: 11000508    	add	w8, w8, #0x1
100000404: b90007e8    	str	w8, [sp, #0x4]
100000408: 17ffffda    	b	0x100000370 <__Z19test_multiple_exitsi+0x10>
10000040c: b9000fff    	str	wzr, [sp, #0xc]
100000410: 14000001    	b	0x100000414 <__Z19test_multiple_exitsi+0xb4>
100000414: b9400fe0    	ldr	w0, [sp, #0xc]
100000418: 910043ff    	add	sp, sp, #0x10
10000041c: d65f03c0    	ret

0000000100000420 <_main>:
100000420: d10083ff    	sub	sp, sp, #0x20
100000424: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000428: 910043fd    	add	x29, sp, #0x10
10000042c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000430: 52800540    	mov	w0, #0x2a               ; =42
100000434: 97ffffcb    	bl	0x100000360 <__Z19test_multiple_exitsi>
100000438: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000043c: 910083ff    	add	sp, sp, #0x20
100000440: d65f03c0    	ret

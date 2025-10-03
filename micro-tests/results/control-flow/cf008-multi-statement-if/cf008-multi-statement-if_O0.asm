
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf008-multi-statement-if/cf008-multi-statement-if_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_multi_statementi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b90007ff    	str	wzr, [sp, #0x4]
10000036c: b9400be8    	ldr	w8, [sp, #0x8]
100000370: 71000108    	subs	w8, w8, #0x0
100000374: 5400016d    	b.le	0x1000003a0 <__Z20test_multi_statementi+0x40>
100000378: 14000001    	b	0x10000037c <__Z20test_multi_statementi+0x1c>
10000037c: b9400be8    	ldr	w8, [sp, #0x8]
100000380: 531f7908    	lsl	w8, w8, #1
100000384: b90007e8    	str	w8, [sp, #0x4]
100000388: b94007e8    	ldr	w8, [sp, #0x4]
10000038c: 11002908    	add	w8, w8, #0xa
100000390: b90007e8    	str	w8, [sp, #0x4]
100000394: b94007e8    	ldr	w8, [sp, #0x4]
100000398: b9000fe8    	str	w8, [sp, #0xc]
10000039c: 14000004    	b	0x1000003ac <__Z20test_multi_statementi+0x4c>
1000003a0: 12800008    	mov	w8, #-0x1               ; =-1
1000003a4: b9000fe8    	str	w8, [sp, #0xc]
1000003a8: 14000001    	b	0x1000003ac <__Z20test_multi_statementi+0x4c>
1000003ac: b9400fe0    	ldr	w0, [sp, #0xc]
1000003b0: 910043ff    	add	sp, sp, #0x10
1000003b4: d65f03c0    	ret

00000001000003b8 <_main>:
1000003b8: d10083ff    	sub	sp, sp, #0x20
1000003bc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c0: 910043fd    	add	x29, sp, #0x10
1000003c4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c8: 528000a0    	mov	w0, #0x5                ; =5
1000003cc: 97ffffe5    	bl	0x100000360 <__Z20test_multi_statementi>
1000003d0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d4: 910083ff    	add	sp, sp, #0x20
1000003d8: d65f03c0    	ret


/Users/jim/work/cppfort/micro-tests/results/control-flow/cf077-return-in-loop/cf077-return-in-loop_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_return_in_loopi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b90007ff    	str	wzr, [sp, #0x4]
10000036c: 14000001    	b	0x100000370 <__Z19test_return_in_loopi+0x10>
100000370: b94007e8    	ldr	w8, [sp, #0x4]
100000374: 71019108    	subs	w8, w8, #0x64
100000378: 540001ea    	b.ge	0x1000003b4 <__Z19test_return_in_loopi+0x54>
10000037c: 14000001    	b	0x100000380 <__Z19test_return_in_loopi+0x20>
100000380: b94007e8    	ldr	w8, [sp, #0x4]
100000384: b9400be9    	ldr	w9, [sp, #0x8]
100000388: 6b090108    	subs	w8, w8, w9
10000038c: 540000a1    	b.ne	0x1000003a0 <__Z19test_return_in_loopi+0x40>
100000390: 14000001    	b	0x100000394 <__Z19test_return_in_loopi+0x34>
100000394: b94007e8    	ldr	w8, [sp, #0x4]
100000398: b9000fe8    	str	w8, [sp, #0xc]
10000039c: 14000009    	b	0x1000003c0 <__Z19test_return_in_loopi+0x60>
1000003a0: 14000001    	b	0x1000003a4 <__Z19test_return_in_loopi+0x44>
1000003a4: b94007e8    	ldr	w8, [sp, #0x4]
1000003a8: 11000508    	add	w8, w8, #0x1
1000003ac: b90007e8    	str	w8, [sp, #0x4]
1000003b0: 17fffff0    	b	0x100000370 <__Z19test_return_in_loopi+0x10>
1000003b4: 12800008    	mov	w8, #-0x1               ; =-1
1000003b8: b9000fe8    	str	w8, [sp, #0xc]
1000003bc: 14000001    	b	0x1000003c0 <__Z19test_return_in_loopi+0x60>
1000003c0: b9400fe0    	ldr	w0, [sp, #0xc]
1000003c4: 910043ff    	add	sp, sp, #0x10
1000003c8: d65f03c0    	ret

00000001000003cc <_main>:
1000003cc: d10083ff    	sub	sp, sp, #0x20
1000003d0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003d4: 910043fd    	add	x29, sp, #0x10
1000003d8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003dc: 52800540    	mov	w0, #0x2a               ; =42
1000003e0: 97ffffe0    	bl	0x100000360 <__Z19test_return_in_loopi>
1000003e4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e8: 910083ff    	add	sp, sp, #0x20
1000003ec: d65f03c0    	ret

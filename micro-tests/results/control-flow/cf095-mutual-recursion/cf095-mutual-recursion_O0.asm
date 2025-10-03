
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf095-mutual-recursion/cf095-mutual-recursion_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z7is_eveni>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000368: 910043fd    	add	x29, sp, #0x10
10000036c: b9000be0    	str	w0, [sp, #0x8]
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 350000a8    	cbnz	w8, 0x100000388 <__Z7is_eveni+0x28>
100000378: 14000001    	b	0x10000037c <__Z7is_eveni+0x1c>
10000037c: 52800028    	mov	w8, #0x1                ; =1
100000380: b81fc3a8    	stur	w8, [x29, #-0x4]
100000384: 14000006    	b	0x10000039c <__Z7is_eveni+0x3c>
100000388: b9400be8    	ldr	w8, [sp, #0x8]
10000038c: 71000500    	subs	w0, w8, #0x1
100000390: 94000007    	bl	0x1000003ac <__Z6is_oddi>
100000394: b81fc3a0    	stur	w0, [x29, #-0x4]
100000398: 14000001    	b	0x10000039c <__Z7is_eveni+0x3c>
10000039c: b85fc3a0    	ldur	w0, [x29, #-0x4]
1000003a0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a4: 910083ff    	add	sp, sp, #0x20
1000003a8: d65f03c0    	ret

00000001000003ac <__Z6is_oddi>:
1000003ac: d10083ff    	sub	sp, sp, #0x20
1000003b0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b4: 910043fd    	add	x29, sp, #0x10
1000003b8: b9000be0    	str	w0, [sp, #0x8]
1000003bc: b9400be8    	ldr	w8, [sp, #0x8]
1000003c0: 35000088    	cbnz	w8, 0x1000003d0 <__Z6is_oddi+0x24>
1000003c4: 14000001    	b	0x1000003c8 <__Z6is_oddi+0x1c>
1000003c8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003cc: 14000006    	b	0x1000003e4 <__Z6is_oddi+0x38>
1000003d0: b9400be8    	ldr	w8, [sp, #0x8]
1000003d4: 71000500    	subs	w0, w8, #0x1
1000003d8: 97ffffe2    	bl	0x100000360 <__Z7is_eveni>
1000003dc: b81fc3a0    	stur	w0, [x29, #-0x4]
1000003e0: 14000001    	b	0x1000003e4 <__Z6is_oddi+0x38>
1000003e4: b85fc3a0    	ldur	w0, [x29, #-0x4]
1000003e8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003ec: 910083ff    	add	sp, sp, #0x20
1000003f0: d65f03c0    	ret

00000001000003f4 <_main>:
1000003f4: d10083ff    	sub	sp, sp, #0x20
1000003f8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003fc: 910043fd    	add	x29, sp, #0x10
100000400: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000404: 52800140    	mov	w0, #0xa                ; =10
100000408: 97ffffd6    	bl	0x100000360 <__Z7is_eveni>
10000040c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000410: 910083ff    	add	sp, sp, #0x20
100000414: d65f03c0    	ret

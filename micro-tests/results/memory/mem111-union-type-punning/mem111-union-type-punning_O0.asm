
/Users/jim/work/cppfort/micro-tests/results/memory/mem111-union-type-punning/mem111-union-type-punning_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_union_punningv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 529eb868    	mov	w8, #0xf5c3             ; =62915
100000368: 72a80908    	movk	w8, #0x4048, lsl #16
10000036c: 1e270100    	fmov	s0, w8
100000370: bd000fe0    	str	s0, [sp, #0xc]
100000374: b9400fe8    	ldr	w8, [sp, #0xc]
100000378: 71000108    	subs	w8, w8, #0x0
10000037c: 1a9f07e0    	cset	w0, ne
100000380: 910043ff    	add	sp, sp, #0x10
100000384: d65f03c0    	ret

0000000100000388 <_main>:
100000388: d10083ff    	sub	sp, sp, #0x20
10000038c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000390: 910043fd    	add	x29, sp, #0x10
100000394: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000398: 97fffff2    	bl	0x100000360 <__Z18test_union_punningv>
10000039c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a0: 910083ff    	add	sp, sp, #0x20
1000003a4: d65f03c0    	ret


/Users/jim/work/cppfort/micro-tests/results/functions/fn023-overload/fn023-overload_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z4funci>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9400fe0    	ldr	w0, [sp, #0xc]
10000036c: 910043ff    	add	sp, sp, #0x10
100000370: d65f03c0    	ret

0000000100000374 <__Z4funcf>:
100000374: d10043ff    	sub	sp, sp, #0x10
100000378: bd000fe0    	str	s0, [sp, #0xc]
10000037c: bd400fe0    	ldr	s0, [sp, #0xc]
100000380: 910043ff    	add	sp, sp, #0x10
100000384: d65f03c0    	ret

0000000100000388 <_main>:
100000388: d10083ff    	sub	sp, sp, #0x20
10000038c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000390: 910043fd    	add	x29, sp, #0x10
100000394: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000398: 528002e0    	mov	w0, #0x17               ; =23
10000039c: 97fffff1    	bl	0x100000360 <__Z4funci>
1000003a0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003a4: 910083ff    	add	sp, sp, #0x20
1000003a8: d65f03c0    	ret

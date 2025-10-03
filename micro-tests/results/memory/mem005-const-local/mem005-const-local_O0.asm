
/Users/jim/work/cppfort/micro-tests/results/memory/mem005-const-local/mem005-const-local_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_const_localv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 52800540    	mov	w0, #0x2a               ; =42
100000368: b9000fe0    	str	w0, [sp, #0xc]
10000036c: 910043ff    	add	sp, sp, #0x10
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: d10083ff    	sub	sp, sp, #0x20
100000378: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000037c: 910043fd    	add	x29, sp, #0x10
100000380: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000384: 97fffff7    	bl	0x100000360 <__Z16test_const_localv>
100000388: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000038c: 910083ff    	add	sp, sp, #0x20
100000390: d65f03c0    	ret

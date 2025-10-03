
/Users/jim/work/cppfort/micro-tests/results/memory/mem080-dangling-reference/mem080-dangling-reference_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18dangling_referencev>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 910033e0    	add	x0, sp, #0xc
100000368: 910043ff    	add	sp, sp, #0x10
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800000    	mov	w0, #0x0                ; =0
100000374: d65f03c0    	ret

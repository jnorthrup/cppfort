
/Users/jim/work/cppfort/micro-tests/results/memory/mem079-reference-return/mem079-reference-return_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z13get_referencev>:
1000003f8: 90000020    	adrp	x0, 0x100004000 <__ZZ13get_referencevE1x>
1000003fc: 91000000    	add	x0, x0, #0x0
100000400: d65f03c0    	ret

0000000100000404 <_main>:
100000404: 90000028    	adrp	x8, 0x100004000 <__ZZ13get_referencevE1x>
100000408: b9400100    	ldr	w0, [x8]
10000040c: d65f03c0    	ret

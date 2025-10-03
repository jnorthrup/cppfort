
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf080-return-reference/cf080-return-reference_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z21test_return_referencev>:
1000003f8: 90000020    	adrp	x0, 0x100004000 <_global>
1000003fc: 91000000    	add	x0, x0, #0x0
100000400: d65f03c0    	ret

0000000100000404 <_main>:
100000404: 52800c88    	mov	w8, #0x64               ; =100
100000408: 90000029    	adrp	x9, 0x100004000 <_global>
10000040c: b9000128    	str	w8, [x9]
100000410: 52800c80    	mov	w0, #0x64               ; =100
100000414: d65f03c0    	ret


/Users/jim/work/cppfort/micro-tests/results/control-flow/cf079-void-return/cf079-void-return_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_void_returniRi>:
100000360: 531f7808    	lsl	w8, w0, #1
100000364: 7100001f    	cmp	w0, #0x0
100000368: 5a9fa108    	csinv	w8, w8, wzr, ge
10000036c: b9000028    	str	w8, [x1]
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 52800280    	mov	w0, #0x14               ; =20
100000378: d65f03c0    	ret

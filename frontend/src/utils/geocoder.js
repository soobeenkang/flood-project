export async function searchAddress(address) {
  return new Promise((resolve, reject) => {
    const geocoder = new window.kakao.maps.services.Geocoder();
    geocoder.addressSearch(address, (result, status) => {
      if (status === window.kakao.maps.services.Status.OK) {
        resolve({ lat: Number(result[0].y), lng: Number(result[0].x) });
      } else {
        reject(new Error("주소 검색 실패"));
      }
    });
  });
}

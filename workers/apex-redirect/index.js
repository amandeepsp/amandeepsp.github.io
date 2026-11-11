// Permanently redirects amandeeps.net and www.amandeeps.net to the blog, keeping path and query.
export default {
    fetch(request) {
        const url = new URL(request.url);
        url.hostname = "blog.amandeeps.net";
        url.protocol = "https:";
        url.port = "";
        return Response.redirect(url.toString(), 301);
    }
};

<?php

namespace App\Controller;

use App\Repository\UserRepository;
use DateInterval;
use Psr\Cache\CacheItemPoolInterface;
use Psr\Cache\InvalidArgumentException;
use Symfony\Bundle\FrameworkBundle\Controller\AbstractController;
use Symfony\Component\Cache\Adapter\AdapterInterface;
use Symfony\Component\Cache\CacheItem;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\Routing\Annotation\Route;

class AdminController extends AbstractController
{

    private CacheItemPoolInterface $cache;

    public function __construct(CacheItemPoolInterface $cache)
    {
        $this->cache = $cache;
    }

    #[Route('/', name: 'app_admin')]
    public function index(): Response
    {
        $this->denyAccessUnlessGranted("IS_AUTHENTICATED_FULLY");


        return $this->render('admin/index.html.twig');
    }

    /**
     * @throws InvalidArgumentException
     */
    #[Route('/redis_test', name: 'redis_test')]
    public function someAction(Request $request, CacheItemPoolInterface $cacheMyRedis, UserRepository $usersRepo)
    {

        /** @var CacheItemPoolInterface $cachedDataItem */
        $cachedDataItem = $cacheMyRedis->getItem('qwe');


    }
}
